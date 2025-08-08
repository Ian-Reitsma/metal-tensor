# metal-tensor

This repository is used as a submodule of `metal-orchard`.

## Working from the parent repo

When editing files here while it is checked out as part of the `metal-orchard`
repository, commit and push your changes **from this submodule first** before
updating the parent repository:

```bash
cd submodules/metal-tensor
# make changes, then commit
git commit -am "Describe change"
git push origin agent/codex

cd ../..
# update the root repository to point to the new commit
git add submodules/metal-tensor
git commit -m "Update metal-tensor submodule pointer"
git push origin agent/codex
```

If the submodule commit is not pushed before the parent commit, later clones of
`metal-orchard` will fail during `git submodule update` because the referenced
commit cannot be fetched. After pushing both repositories, verify from a clean
clone:

```bash
git submodule update --init --recursive
```

