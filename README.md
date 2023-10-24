# GazpreaBase
The base cmake setup for Gazprea assignment.

Author: Braedy Kuzma (braedy@ualberta.ca)  
Updated by: Deric Cheung (dacheung@ualberta.ca)

# Building
CMake will use the environment variables ANTLR_INS, ANTLR_JAR, and MLIR_DIR
to find locally installed builds of ANTLR and MLIR.
See the [setup document](https://webdocs.cs.ualberta.ca/~c415/setup/)
to find out how to install these packages on your machine.

  1. Make a directory that you intend to build the project in and change into
     that directory.
  2. Run `cmake <path-to-GazpreaBase>`.
  3. Run `make`.

# Pulling in upstream changes
If there are updates to your assignment you can retrieve them using the
instructions here.
  1. Add the upstream as a remote using `git remote add upstream <clone-link>`.
  1. Fetch updates from the upstream using `git fetch upstream`
  1. Merge the updates into a local branch using
     `git merge <local branch> upstream/<upstream branch>`. Usually both
     branches are `master`.
  1. Make sure that everything builds before committing to your personal
     master! It's much easier to try again if you can make a fresh clone
     without the merge!

Once the remote has been added, future updates are simply the `fetch` and
`merge` steps.
