## Contributions Model

New contributions to *PelePhysics* are welcome !

The *PelePhysics* contributions workflow follows these steps:
1. Fork the main repository
2. Create an `AmazingNewFeature` branch implementing your changes
3. Open a Pull Request (PR) from `AmazingNewFeature` on your fork to branch `development` of the main *PelePhysics* repository

Follow [GitHub directions](https://docs.github.com/en/free-pro-team@latest/github/getting-started-with-github/fork-a-repo)
to fork *PelePhysics* main repo on your GitHub account, and use a recursive `git clone` to get your fork of *PelePhysics* and its dependencies.

Then step into the *PelePhysics* folder and add the main *PelePhysics* repository as the `upstream` remote in order to keep track of the main repo :
~~~
git add remote upstream https://github.com/Pele-Suite/PelePhysics
~~~

At any point, you can update the `development` branch of your local repository with changes implemented in the main *PelePhysics* repo by pulling from `upstream` :
~~~
git checkout development
git pull upstream development
~~~

We recommend setting your development branch to track the upstream one instead of your fork:
~~~
git branch -u upstream/development
~~~

You are now free to modify your own fork of *PelePhysics*. To add a new feature to *PelePhysics*, the procedure is:

1. Create a branch for the new feature from the `development` branch (locally) :
   ~~~
   git checkout development
   git checkout -b AmazingNewFeature
   ~~~
2. and commit your changes to your local repo :
   ~~~
   git commit -m "Developed AmazingNewFeature"
   ~~~
3. Alongside your development, regularly merge changes from the main repo `development` branch into your `AmazingNewFeature` branch,
fix any conflicts, and push your changes to your GitHub fork :
   ~~~
   git push -u origin AmazingNewFeature
   ~~~
4. When you are ready to propose your new feature/improvement/bug fix to the main *PelePhysics* repo, reiterate Step 3 and submit a PR through the GitHub page from your fork onto the `development` branch of the main repo:

 - Click on the ``compare & pull request`` button to start your PR.
 - Provide a title and a short description for your PR:
   * what feature/fix do you propose
   * how did you test it
   * any other information deemed useful : does it modify the default *PeleLM* behavior ? ...
 - Press ``Create pull request``.

Please DO NOT write large PR, as they are very difficult and time-consuming to review.
As much as possible, split them into small targeted PRs.
For example, if find typos in the documentation open a pull request that only fixes typos.
If you want to fix a bug, make a small pull request that only fixes a bug.

## PelePhysics Coding Style Guide

Source code files can be automatically formatted to adhere to the appropriate formatting rules using ``clang-format``. To format all files, use the command:
~~~
find Source Testing \( -name "*.cpp" -o -name "*.H" \) -exec clang-format -i {} +
~~~

from within the PelePhysics base directory. You can also format files individually using ``clang-format -i /path/to/file``. Adherence to this format is checked for all PRs.

Beyond that, as much as possible, `PelePhysics` adheres to [AMReX Coding Style](https://github.com/AMReX-Codes/amrex/blob/development/CONTRIBUTING.md#amrex-coding-style-guide)
and we are encouraging contributors to follow those guidelines.
