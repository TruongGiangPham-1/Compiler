# LordFarquaad
## Introduction
A compiler for a language(developed by IBM) using C++, LLVM and MLIR (LLVM-dialect). The specifications of the language can be found [here](https://cmput415.github.io/415-docs/gazprea/).

## Authors
The base cmake setup by

Author: Braedy Kuzma (braedy@ualberta.ca)  

Updated by: Deric Cheung (dacheung@ualberta.ca)

Updated by: Quinn Pham (qpham@ualberta.ca)

Project done by:    Joshua Ji, Stan Fedyk, Truong-Giang Pham, Sanjeev Kotha
                    
# Usage
## Installing MLIR
Due to the complex nature (and size) of the project we did not want to include
MLIR as a subproject. Therefore, there is some additional setup required to get
your build up and running.

### On a personal machine
Follow the instructions on the
     [setup page](https://cmput415.github.io/415-docs/setup/cs_computers.html)
     for your machine.

## Building
### Linux
  1. Install git, java (only the runtime is necessary), and cmake (>= v3.0).
     - Until now, cmake has found the dependencies without issues. If you
       encounter an issue, let a TA know and we can fix it.
  1. Make a directory that you intend to build the project in and change into
     that directory.
  1. Run `cmake <path-to-cmake-src>`.
  1. Run `make`.
  1. Done.
It should produce a `bin` folder.

## Compile and run a program
  1. `./bin/gazc <path-to-input-file>.in <path-to-output-file>.ll` to compile the input program into LLVM IR
  2.  `lli -dlopen=./bin/libgazrt.so <path-to-output-file>.ll` to execute the LLVM IR program with the dynamic library

## Pulling in upstream changes
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

# How it work
  1. grammar written in ANTLR4 which parses the program to create a parse tree
  2. the parse tree is converted to AST tree, which removes uneccesary nodes.
  3. a define + reference pass to the AST to resolve symbols and scopes.
  4. a typecheck pass to the AST to  resolve, promote, and infer types.
  5. a syntax check pass to the AST to raise more compile time errors.
  6. code generation pass to emit LLVM IR.



