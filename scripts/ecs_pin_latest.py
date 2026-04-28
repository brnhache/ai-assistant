#!/usr/bin/env python3
"""Pin the just-pushed `:latest` ECR image into a fresh ECS task definition revision and roll the service.

Why this exists
---------------
`aws ecs update-service --force-new-deployment` recreates tasks but does NOT force
the host to re-pull `:latest`. That means a stale image can serve traffic forever
even after a successful `docker push`. The robust fix is to pin the task
definition to an explicit image digest (`...@sha256:...`) for every roll.

Usage
-----
Run AFTER `aipush` (or any `docker push` to the same repo). Defaults match the
desert-ai-assistant staging service; override with env vars if needed.

    python scripts/ecs_pin_latest.py            # roll service
    python scripts/ecs_pin_latest.py --dry-run  # show what it would do

Env / flags
-----------
    --cluster          (default: default)
    --service          (default: desert-ai-assistant-1fd2)
    --repo             (default: desert-ai-assistant)
    --tag              (default: latest)              ECR tag whose digest we pin
    --region           (default: us-east-1)
    --wait             (default: 300)                 seconds to wait for COMPLETED
    --dry-run

Requirements
------------
    pip / system: awscli, jq is NOT required (we use boto3-equivalent JSON via aws CLI)
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from typing import Any


_QUIET = False


def run(cmd: list[str], capture: bool = True, check: bool = True, quiet: bool = False) -> str:
    """Run a command, return stdout. Raises on nonzero unless check=False."""
    printable = " ".join(shlex.quote(p) for p in cmd)
    if not (quiet or _QUIET):
        print(f"$ {printable}", file=sys.stderr)
    proc = subprocess.run(cmd, capture_output=capture, text=True, check=False)
    if check and proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise SystemExit(f"command failed (exit {proc.returncode}): {printable}")
    return proc.stdout


def aws_json(cmd: list[str], quiet: bool = False) -> Any:
    out = run(cmd, quiet=quiet)
    return json.loads(out) if out.strip() else None


def get_pushed_digest(repo: str, tag: str, region: str) -> str:
    data = aws_json(
        [
            "aws", "ecr", "describe-images",
            "--repository-name", repo,
            "--image-ids", f"imageTag={tag}",
            "--region", region,
            "--output", "json",
        ]
    )
    details = data.get("imageDetails", []) if isinstance(data, dict) else []
    if not details:
        raise SystemExit(f"no image found in ECR repo={repo} tag={tag}")
    digest = details[0]["imageDigest"]
    return digest


def get_current_taskdef_arn(cluster: str, service: str, region: str) -> str:
    data = aws_json(
        [
            "aws", "ecs", "describe-services",
            "--cluster", cluster,
            "--services", service,
            "--region", region,
            "--output", "json",
        ]
    )
    if not isinstance(data, dict):
        raise SystemExit(f"unexpected describe-services response: {data!r}")
    failures = data.get("failures") or []
    services = data.get("services") or []
    if failures and not services:
        raise SystemExit(f"describe-services failures: {json.dumps(failures)}")
    if not services:
        raise SystemExit(
            f"service not found: cluster={cluster} service={service}. "
            f"Raw response keys: {list(data.keys())}"
        )
    svc = services[0]
    td = svc.get("taskDefinition")
    if not td:
        # Dump what we DID get so we can see why.
        print(
            "describe-services returned a service entry without 'taskDefinition'.",
            file=sys.stderr,
        )
        print("Service keys present: " + ", ".join(sorted(svc.keys())), file=sys.stderr)
        print("Status: " + str(svc.get("status")), file=sys.stderr)
        print("Deployments: " + json.dumps(svc.get("deployments", []), default=str)[:1000], file=sys.stderr)
        # Try fallbacks
        deployments = svc.get("deployments") or []
        for d in deployments:
            cand = d.get("taskDefinition")
            if cand:
                print(f"Falling back to deployment taskDefinition: {cand}", file=sys.stderr)
                return cand
        raise SystemExit("could not determine current task definition arn")
    return td


def get_taskdef(td_arn: str, region: str) -> dict:
    data = aws_json(
        [
            "aws", "ecs", "describe-task-definition",
            "--task-definition", td_arn,
            "--region", region,
            "--output", "json",
        ]
    )
    return data["taskDefinition"]


def get_account_id(region: str) -> str:
    data = aws_json(["aws", "sts", "get-caller-identity", "--region", region, "--output", "json"])
    return data["Account"]


def get_running_digest(cluster: str, service: str, region: str) -> str | None:
    arns = aws_json(
        [
            "aws", "ecs", "list-tasks",
            "--cluster", cluster,
            "--service-name", service,
            "--region", region,
            "--output", "json",
        ]
    )
    task_arns = arns.get("taskArns", []) if isinstance(arns, dict) else []
    if not task_arns:
        return None
    data = aws_json(
        [
            "aws", "ecs", "describe-tasks",
            "--cluster", cluster,
            "--tasks", *task_arns,
            "--region", region,
            "--output", "json",
        ]
    )
    tasks = data.get("tasks", []) if isinstance(data, dict) else []
    for t in tasks:
        for c in t.get("containers", []):
            d = c.get("imageDigest")
            if d:
                return d
    return None


# Fields ECS will reject on register-task-definition (read-only metadata)
STRIP_FIELDS = {
    "taskDefinitionArn",
    "revision",
    "status",
    "requiresAttributes",
    "compatibilities",
    "registeredAt",
    "registeredBy",
    "deregisteredAt",
    "enableFaultInjection",
}


def build_new_taskdef(td: dict, new_image: str) -> dict:
    out = {k: v for k, v in td.items() if k not in STRIP_FIELDS}
    cdefs = out.get("containerDefinitions", [])
    if not cdefs:
        raise SystemExit("task definition has no containerDefinitions")
    # Patch every container that points at the same repo (defensive: usually 1).
    repo_part = new_image.split("@", 1)[0]
    patched = 0
    for c in cdefs:
        cur = c.get("image", "")
        cur_repo = cur.split("@", 1)[0].split(":", 1)[0]
        if cur_repo == repo_part.split(":", 1)[0]:
            c["image"] = new_image
            patched += 1
    if patched == 0:
        # fallback: patch first container regardless
        cdefs[0]["image"] = new_image
    return out


def register_taskdef(td: dict, region: str) -> str:
    payload = json.dumps(td)
    proc = subprocess.run(
        [
            "aws", "ecs", "register-task-definition",
            "--cli-input-json", payload,
            "--region", region,
            "--output", "json",
        ],
        capture_output=True, text=True, check=False,
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise SystemExit("register-task-definition failed")
    data = json.loads(proc.stdout)
    return data["taskDefinition"]["taskDefinitionArn"]


def update_service(cluster: str, service: str, td_arn: str, region: str) -> None:
    aws_json(
        [
            "aws", "ecs", "update-service",
            "--cluster", cluster,
            "--service", service,
            "--task-definition", td_arn,
            "--force-new-deployment",
            "--region", region,
            "--output", "json",
        ]
    )


def wait_for_rollout(
    cluster: str,
    service: str,
    region: str,
    max_seconds: int,
    expected_digest: str,
) -> bool:
    """Wait until a running task is on the expected image digest.

    Why we don't trust rolloutState alone: ECS sometimes leaves the PRIMARY
    deployment's rolloutState as IN_PROGRESS even when desiredCount == runningCount
    and tasks are healthy (we've seen this when an old deployment drains slowly
    or when the service uses certain ALB configurations). The unambiguous
    success signal is: ``the running task is on the new image digest`` AND
    ``desiredCount == runningCount > 0`` for the PRIMARY deployment.

    We also bail early on FAILED rolloutState or non-zero failedTasks.
    """
    start = time.time()
    last_status: tuple | None = None
    last_print = 0.0
    while time.time() - start < max_seconds:
        data = aws_json(
            [
                "aws", "ecs", "describe-services",
                "--cluster", cluster,
                "--services", service,
                "--region", region,
                "--output", "json",
            ],
            quiet=True,
        )
        services = data.get("services", []) if isinstance(data, dict) else []
        if not services:
            time.sleep(5)
            continue
        deployments = services[0].get("deployments", [])
        primary = next((d for d in deployments if d.get("status") == "PRIMARY"), None)
        active_count = sum(1 for d in deployments if d.get("status") == "ACTIVE")

        # Watch for explicit failure on the primary.
        if primary and primary.get("rolloutState") == "FAILED":
            elapsed = int(time.time() - start)
            print(
                f"  [{elapsed:>3}s] rollout: FAILED  reason={primary.get('rolloutStateReason')!r}",
                file=sys.stderr,
            )
            return False

        running_digest = get_running_digest(cluster, service, region)

        primary_state = primary.get("rolloutState") if primary else None
        primary_running = primary.get("runningCount") if primary else None
        primary_desired = primary.get("desiredCount") if primary else None
        primary_failed = primary.get("failedTasks") if primary else None

        elapsed = int(time.time() - start)
        status = (
            primary_state,
            primary_running,
            primary_desired,
            active_count,
            (running_digest or "")[:19],
        )
        now = time.time()
        if status != last_status or now - last_print >= 15:
            digest_short = (running_digest or "none")[:19]
            print(
                f"  [{elapsed:>3}s] rollout={primary_state} "
                f"running={primary_running}/{primary_desired} "
                f"failed={primary_failed} active_deployments={active_count} "
                f"digest={digest_short}\u2026",
                file=sys.stderr,
            )
            last_status = status
            last_print = now

        # Real success signal: the running task is on the new digest AND the
        # primary deployment thinks all desired tasks are running.
        if (
            running_digest == expected_digest
            and primary
            and primary_desired
            and primary_running == primary_desired
            and primary_running > 0
        ):
            return True

        # ECS-reported success is also acceptable (defence in depth).
        if primary_state == "COMPLETED":
            return True

        time.sleep(5)
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cluster", default="default")
    ap.add_argument("--service", default="desert-ai-assistant-1fd2")
    ap.add_argument("--repo", default="desert-ai-assistant")
    ap.add_argument("--tag", default="latest")
    ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--wait", type=int, default=300, help="seconds to wait for rollout")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    account = get_account_id(args.region)
    print(f"account={account} region={args.region}", file=sys.stderr)

    digest = get_pushed_digest(args.repo, args.tag, args.region)
    new_image = f"{account}.dkr.ecr.{args.region}.amazonaws.com/{args.repo}@{digest}"
    print(f"pushed digest:  {digest}", file=sys.stderr)
    print(f"new image:      {new_image}", file=sys.stderr)

    running = get_running_digest(args.cluster, args.service, args.region)
    print(f"running digest: {running or '(none / no tasks)'}", file=sys.stderr)
    if running == digest:
        print("\nAlready running the latest digest. Nothing to do.", file=sys.stderr)
        return 0

    td_arn = get_current_taskdef_arn(args.cluster, args.service, args.region)
    print(f"current taskdef: {td_arn}", file=sys.stderr)
    td = get_taskdef(td_arn, args.region)
    new_td = build_new_taskdef(td, new_image)

    if args.dry_run:
        print("\n--- DRY RUN: would register this task definition ---", file=sys.stderr)
        print(json.dumps(new_td, indent=2, default=str))
        return 0

    new_arn = register_taskdef(new_td, args.region)
    print(f"registered:     {new_arn}", file=sys.stderr)

    update_service(args.cluster, args.service, new_arn, args.region)
    print("rolling service...", file=sys.stderr)

    ok = wait_for_rollout(
        args.cluster, args.service, args.region, args.wait, expected_digest=digest
    )
    if not ok:
        print("\n❌ rollout did not reach COMPLETED in time. Check CloudWatch + ECS events.", file=sys.stderr)
        return 2

    final_running = get_running_digest(args.cluster, args.service, args.region)
    print(f"\n✅ rollout COMPLETED", file=sys.stderr)
    print(f"   running digest now: {final_running}", file=sys.stderr)
    if final_running != digest:
        print(f"   ⚠️  digest mismatch! expected {digest}", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
