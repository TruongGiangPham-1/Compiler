#!/usr/bin/env bash
# CMPUT 415 LLVM configure script
# Author: Braedy Kuzma (braedy AT ualberta.ca)
# This script is designed to make sure that you have an llvm install compatible
# with what we'll be marking you against. I recommend editing this file and
# committing your changes so you guys have identical environments (helps with
# peer support). There's nothing wrong with having diverging environments and
# it doesn't affect me because I won't be using your version.

# http://llvm.org/docs/CMake.html#llvm-specific-variables
# The above link has many options available in it. Here are some relevant ones
# chosen rather hastily that you might care about.
# LLVM_BUILD_TOOLS:
#   Builds all of the tools by default in the "all" target. I've disabled it
#   here so as to not clutter your build.
# LLVM_ENABLE_ASSERTIONS:
#   Potential slow down, usually disabled in non-debug builds. I've enabled
#   them permanently because they're useful for catching things. Feel free to
#   disable if you want, but I'll be running with them on.
# LLVM_ENABLE_RTTI:
#   Potential slow down, disabled in all builds by default. Enables run time
#   type information. (e.g. dynamic_cast, typeid). Feel free to enable this if
#   you need it for a bit of debugging but I'll be running with it off. See
#   here: https://en.wikipedia.org/wiki/Run-time_type_information
# LLVM_PARALLEL_COMPILE_JOBS:
#   Defines the maximum number of parallel compiling threads. Useful if your
#   computer can't handle the compilation (linking is usually the issue and I'm
#   not sure if this overrides the -j argument to make)
# LLVM_PARALLEL_LINK_JOBS:
#   Defines the maximum number of parallel linking threads. Useful if your
#   computer can't handle all of the linking (if your computer is lagging/
#   hanging, linking is probably the issue).
# LLVM_ENABLE_DOXYGEN:
#   Enables local docs building. Requires doxygen?
LLVM_OPTIONS="-DLLVM_BUILD_TOOLS=ON -DLLVM_BUILD_TESTS=OFF -DLLVM_INCLUDE_TESTS=OFF -DLLVM_BUILD_EXAMPLES=OFF -DLLVM_INCLUDE_EXAMPLES=OFF -DLLVM_ENABLE_ASSERTIONS=ON -DLLVM_OPTIMIZED_TABLEGEN=ON"

# -------------------------
# STEP 1: CHOOSE PATHS.
# Ideally these are absolute paths so you can run this script from anywhere.
# Read the note below the sudo trap (the trap is the whole if statement)
# about how the install directory will be used.
BAS_DIR="$HOME/llvm-project"
SRC_DIR="$BAS_DIR/llvm" # The directory inside where you've checked
                                  # out llvm.
BLD_DIR="$BAS_DIR/build" # The directory llvm will be built to.
INS_DIR="$LLVM_INS" # DON'T TOUCH THIS. SEE README.

# Sudo trap.
# This is here because I don't want you creating files as root and possibly
# mucking up something. I don't want to be responsible for that.
if [ "$EUID" -eq 0 ]
  then echo "Don't run me as root. I create files you want to be able to delete."
  exit
fi

mkdir -p "$BLD_DIR"
cd "$BLD_DIR"

# Personally I'm not a fan of installing this to my / partition because it can
# be large (debug can be >20GB depending on options) and cleanup is hard due to
# the lack of an uninstall target and root permissions. I've included the
# ability to use a custom install destination (INS_DIR -> LLVM_DIR) prefix for
# the llvm installation and one that uses the default prefix (i.e.
# /usr/{bin,lib}). The first one will not require root permissions (you own
# the install directory) but requires additional setup for your own project.
# The second one should install the required files somewhere that cmake can
# automatically find them in your personal project.

# -------------------------
# STEP 2: CHOOSE BUILD TYPE.
# UNCOMMENT ONLY ONE BUILD TYPE. I recommend release.

# DEBUG BUILD (full debug build for use with debugger, quite large and
# unoptimised).
# BUILD="DEBUG"

# MINSIZEREL BUILD (minimum size release build, faster, but prioritises small
# size).
# BUILD="MINSIZEREL"

# RELWITHDEBINFO BUILD (builds optimised code with debug info, gives the
# debugger a chance but whole section can be optimised away).
# BUILD="RELWITHDEBINFO"

# RELEASE BUILD (builds optimised code, you can still use this with a debugger
# but any time you step into llvm the compiler will be lost until you return).
BUILD="RELEASE"

# -------------------------
# STEP 3: CHOOSE INSTALL METHOD.
# If you want to install to system directories you will need to swap the comment
# below. This is at *your own risk* and is truly not necessary.
USE_INSTALL="-DCMAKE_INSTALL_PREFIX=$INS_DIR"
# USE_INSTALL=""

# -------------------------
# STEP 4: IGNORE THIS.
# These are things that need to be done without you thinking about it.

# If we're on MacOS, we can't use the gold linker. It's an ELF-only tool.
USE_GOLD=""
unamestr=`uname`
if [[ "$unamestr" == 'Linux' ]]; then
   USE_GOLD="-DLLVM_USE_LINKER=gold"
fi

cmake "$SRC_DIR" -DCMAKE_BUILD_TYPE="$BUILD" -DLLVM_TARGETS_TO_BUILD=X86 $USE_GOLD $USE_INSTALL $LLVM_OPTIONS
