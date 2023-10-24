# GazpreaBase
The base cmake setup for Gazprea assignment.

Author: Braedy Kuzma (braedy@ualberta.ca)  
Updated by: Deric Cheung (dacheung@ualberta.ca)
Updated by: Quinn Pham (qpham@ualberta.ca)

# Usage
## Installing LLVM
In this project, you will be working with LLVM. Due to the complex nature (and
size) of the project, we did not want to include LLVM as a subproject.
Therefore, there is some additional setup required to get your build up and
running.

### On a personal machine
The first thing you should do is have a look into the `configureLLVM.sh` script
in the scripts folder and understand as much as possible. We'll touch some
things briefly here that are better explained there. The steps here will expect
you have some knowledge of what's going on inside the script.

  1. Clone the [llvm project](https://github.com/llvm/llvm-project) into your
     home directory, checkout the 15.0.7 release, and change `configureLLVM.sh`
     as needed.
     1. `cd $HOME`
     2. `git clone git@github.com:llvm/llvm-project.git`
     3. `cd llvm-project`
     4. `git checkout llvmorg-15.0.7`
  2. Add these configuration lines to your `~/.bashrc` on linux or
     `~/.zshenv` on macOS to setup the next steps. You should restart your
     terminal after editing these files.
      ```bash
      export LLVM_INS="$HOME/llvm-ins/"           # Change me if you want.
      export LLVM_DIR="$LLVM_INS/lib/cmake/llvm/" # Don't change me.
      export PATH="$LLVM_INS/bin:$PATH"           # Don't change me.
      ```
  3. If you want a debug build you may change this by looking at Step 2 in the
     `scripts/configureLLVM.sh` script.
  4. Run `configureLLVM.sh`.
  5. `cd $HOME/llvm-project/build`
  6. `make -j x` where x is the number of compilation threads. `4` is safe if
     you have >= 8gb RAM. Haven't experimented much here. The script is set up
     to try to use a better linker, but if you end up with your system linker
     the memory usage can balloon quickly and paging can become a problem. If
     you're having problems with hanging or system lag while compiling you
     might want to kill this (you won't lose your progress if `ctrl+c` works)
     and run with less threads.
  7. Run `make install`.
  8. CMake should automatically pick up your built llvm now. You should return
     to the assignment specification to complete setup.

### On a university machine
You won't be building LLVM on the university machines: AICT wouldn't be very
happy with you. Instead, we are providing a **RELEASE** build available for
everyone.
  1. Follow the instructions on the
     [setup page](https://cmput415.github.io/415-docs/setup/cs_computers.html)
     for the CS computers and LLVM will be available to you.

## Building
### Linux
  1. Install git, java (only the runtime is necessary), and cmake (>= v3.0).
     - Until now, cmake has found the dependencies without issues. If you
       encounter an issue, let a TA know and we can fix it.
  2. Make a directory that you intend to build the project in and change into
     that directory.
  3. Run `cmake <path-to-Gazprea-Base>`.
  4. Run `make`.
  5. Done.

## Pulling in upstream changes
If there are updates to your assignment you can retrieve them using the
instructions here.
  1. Add the upstream as a remote using `git remote add upstream <clone-link>`.
  2. Fetch updates from the upstream using `git fetch upstream`
  3. Merge the updates into a local branch using
     `git merge <local branch> upstream/<upstream branch>`. Usually both
     branches are `master`.
  4. Make sure that everything builds before committing to your personal
     master! It's much easier to try again if you can make a fresh clone
     without the merge!

Once the remote has been added, future updates are simply the `fetch` and
`merge` steps.
