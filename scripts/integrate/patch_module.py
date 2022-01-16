#!/usr/bin/env python
# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Sets a submodule state to patched or normal, allowing side development on
# the submodule that proceeds ahead of the upstream repository.
#
# This is an advanced feature, and it's only recommended use is to make:
#   a) Short-lived changes to an upstream repo necessary to fix a release, etc.
#   b) Integrate cherry-picks that have already landed in the upstream repo
#      and that a future integrate will therefore pick up as part of normal
#      operations.
#
# Effectively, this lets you push the development process where you commit
# local changes to a submodule to the team and project, switching everyone
# to a module branch that is running ahead of upstream main.
#
# The best time to change a module state to 'patched' is before making a change
# when both the parent repo and the submodule are pristine. Various more
# advanced flows are possible, but you have been warned. You are unlikely to
# mess something up permanently or lose work, but you will almost certainly
# lose time.
#
# Using third_party/llvm-project as an example, here is a supported flow:
#
# 1. Upstream cherry-pick is identified which we would like merged into the
#    project ahead of an integrate.
# 2. Switch the module to 'patched':
#      patch_module.py --module=llvm-project
# 3. Push the resultant commit to the main repository. No code has been changed
#    at this point, just submodule metadata. There will be a branch created
#    in iree-llvm-fork with a name like patched-llvm-project-YYYYMMDD[.n] and
#    your local llvm-project submodule will have a 'patched' remote and will
#    be switched to a tracking branch of this remote.
# 4. Team members who need to contribute to the patched branch:
#      a. Pull from the main repo and verify that the repository is in a
#         patched state (check for third_party/llvm-project.branch-pin):
#           patch_module.py --module=llvm-project --command=status
#      b. Run:
#           patch_module.py --module=llvm-project
#         This will configure their local repository to track the patch
#         branch, allowing commits to be pushed.
#      c. Run `git cherry-pick`, etc in the llvm-project submodule.
#      d. Run `git push` in the llvm-project submodule to make cherry-pick
#         commits available to everyone.
#      e. Continue main project development and commits as usual.
# 5. Someone eventually does an LLVM bump, which unpins the patch branch
#    and reverts to upstream head. If only cherry-picks were done, all
#    upstream commits should be incorporated, and the patching process now
#    concludes until someone wants to switch back to the 'patched' state.

import argparse
from datetime import date
import os
import sys

import iree_utils
import iree_modules

PATCH_REMOTE_ALIAS = "patched"


def main(args):
    module_info = iree_modules.MODULE_INFOS.get(args.module)
    if not module_info:
        raise SystemExit(f"ERROR: Bad value for --module. Must be one of: "
                         f"{', '.join(iree_modules.MODULE_INFOS.keys())}")

    if args.command == "patch":
        main_patch(args, module_info)
    elif args.command == "unpatch":
        main_unpatch(args, module_info)
    elif args.command == "status":
        main_status(args, module_info)
    else:
        raise SystemExit(
            f"ERROR: Unrecognized --command. Must be one of: patch, unpatch")


def main_patch(args, module_info: iree_modules.ModuleInfo):
    branch_pin_file = os.path.join(iree_utils.get_repo_root(),
                                   module_info.branch_pin_file)
    module_root = os.path.join(iree_utils.get_repo_root(), module_info.path)
    setup_module_remotes(module_root, module_info)

    # If the module is not already patched, setup a new branch on the server
    # and push to it.
    if not os.path.exists(branch_pin_file):
        iree_utils.git_check_porcelain()
        print(
            f"Module {module_info.name} is not patched. Creating a new patch branch."
        )
        branch_name = find_unused_branch_name(module_info)
        print(f"Allocated branch: {branch_name}")
        current_commit, summary = iree_utils.git_current_commit(
            repo_dir=module_root)
        print(f"Module is currently at: {summary}")
        print(
            f"*** Pushing branch {branch_name} to {module_info.fork_repository_push} ***"
        )
        print(f"(Please ignore any messages below about creating a PR)\n")
        iree_utils.git_exec([
            "push", PATCH_REMOTE_ALIAS,
            f"{current_commit}:refs/heads/{branch_name}"
        ],
                            repo_dir=module_root)
        print(f"*** Branch {branch_name} pushed ***")
        iree_utils.git_submodule_set_origin(
            module_info.path,
            url=module_info.fork_repository_pull,
            branch="--default")
        with open(branch_pin_file, "wt") as f:
            print(branch_name, file=f)
        # Add files.
        iree_utils.git_exec(
            ["add", module_info.branch_pin_file, ".gitmodules"])
        iree_utils.git_create_commit(message=(
            f"Pin {module_info.path} to {branch_name}\n"
            f"  * Update submodule url to {module_info.fork_repository_pull}\n"
            f"  * Add pinned state"))

    # Ok, now that we know the server is set up, put the submodule on the right
    # branch and make it track so that people can just push directly from there.
    with open(branch_pin_file, "rt") as f:
        branch_name = f.read().strip()
    print(
        f"Checking out remote patch branch {branch_name} in {module_info.path}."
    )
    iree_utils.git_fetch(repository=PATCH_REMOTE_ALIAS,
                         ref=branch_name,
                         repo_dir=module_root)
    if not iree_utils.git_branch_exists(branch_name, repo_dir=module_root):
        print(f"Setting up local branch {branch_name}")
        iree_utils.git_create_branch(branch_name,
                                     ref=f"{PATCH_REMOTE_ALIAS}/{branch_name}",
                                     repo_dir=module_root)

    # Switch to the branch.
    print(f"If you have made commits to this module prior to setting up the ")
    print(
        f"patch, they will not be automatically applied. Recommend inspecting")
    print(f"`git reflog` and porting them in once this step completes.")
    if iree_utils.git_is_porcelain(repo_dir=module_root):
        iree_utils.git_exec(["switch", branch_name], repo_dir=module_root)
    else:
        print(f"*** YOU HAVE CHANGES IN {module_root} ***")
        print(f"Figure out what to do with those and then run:")
        print(f"  git switch {branch_name}")

    print(f"******* Congratulations *******")
    print(
        f"Your module is now in the 'patched' state. You can commit changes ")
    print(
        f"locally. You must push commits to the module patch branch prior to ")
    print(f"the main repo:")
    print(f"  (cd {module_info.path} && git push patched)")
    print(f"Run this script with --command=unpatch to return to upstream head")
    print(
        f"Module metadata changes have been committed to the main repository ")
    print(f"and can conflict with others deciding to patch this module.")
    print(f"Recommend pushing the main repo to secure the lock.")


def main_unpatch(args, module_info: iree_modules.ModuleInfo):
    branch_pin_file = os.path.join(iree_utils.get_repo_root(),
                                   module_info.branch_pin_file)
    module_root = os.path.join(iree_utils.get_repo_root(), module_info.path)

    if not os.path.exists(branch_pin_file):
        print(f"Module does not appear to be patched: No {branch_pin_file}")
        return

    # Rollback the module state changes.
    iree_utils.git_submodule_set_origin(module_info.path,
                                        url=module_info.default_repository_url,
                                        branch="--default")
    os.remove(branch_pin_file)
    iree_utils.git_exec(["add", module_info.branch_pin_file, ".gitmodules"])
    print("******* Please Read *******")
    print(
        f"We have reset the metadata for module {module_info.path} to track ")
    print(f"the upstream repository. However, we have no way of knowing what ")
    print(f"upstream commit to pin the module to. You must arrange this ")
    print(f"by entering {module_info.path}, selecting an appropriate commit ")
    print(f"and committing a change to the main repository.")
    print(f"The most sure way to do this is to select a commit that includes ")
    print(f"all landed changes and then make any needed fixes. Or proceed ")
    print(f"with an established integration procedure.")


def main_status(args, module_info: iree_modules.ModuleInfo):
    branch_pin_file = os.path.join(iree_utils.get_repo_root(),
                                   module_info.branch_pin_file)
    # Check pin file.
    if os.path.exists(branch_pin_file):
        with open(branch_pin_file, "rt") as f:
            branch_name = f.read().strip()
        print(f"MODULE PINNED TO PATCH BRANCH: {branch_name}")
        print(
            f"  (it is safe to run patch_module.py to configure your local repo)"
        )
    else:
        print(f"MODULE NOT PINNED TO A PATCH")


def setup_module_remotes(module_root: str,
                         module_info: iree_modules.ModuleInfo):
    iree_utils.git_setup_remote(PATCH_REMOTE_ALIAS,
                                url=module_info.fork_repository_push,
                                repo_dir=module_root)


def find_unused_branch_name(module_info: iree_modules.ModuleInfo):
    branch_base = f"{module_info.branch_prefix}{date.today().strftime('%Y%m%d')}"
    branch_name = branch_base
    existing_branches = iree_utils.git_ls_remote_branches(
        module_info.fork_repository_pull,
        filter=[f"refs/heads/{module_info.branch_prefix}*"])
    i = 1
    while branch_name in existing_branches:
        branch_name = f"{branch_base}.{i}"
        i += 1
    return branch_name


def parse_arguments(argv):
    parser = argparse.ArgumentParser(description="IREE Submodule Patcher")
    parser.add_argument("--module",
                        help="Submodule to operate on",
                        default=None)
    parser.add_argument("--command",
                        help="Command to execute",
                        default="patch")
    args = parser.parse_args(argv)
    return args


if __name__ == "__main__":
    main(parse_arguments(sys.argv[1:]))
