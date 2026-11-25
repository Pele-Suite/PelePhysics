#!/usr/bin/env bash
help()
{
   # Display Help
   echo "Convert mechanism yaml file."
   echo
   echo "Syntax: converter.sh [-h|f]"
   echo "options:"
   echo "h     Print this Help."
   echo "f     Convert mechanism yaml file."
   echo
}

function abspath() {
    if which realpath > /dev/null; then
        if [ -d "$1" ]; then
            echo "$(realpath "$1")";
        else
            echo "$(realpath .)";
        fi;
    else
        pushd . > /dev/null;
        if [ -d "$1" ]; then
            cd "$1" || exit;
            dirs -l +0;
        else
            cd "$(dirname "$1")" || exit;
            cur_dir=$(dirs -l +0);
            if [ "$cur_dir" == "/" ]; then
                echo "$cur_dir$(basename "$1")";
            else
                echo "$cur_dir/$(basename "$1")";
            fi;
        fi;
        popd > /dev/null || exit;
    fi;
}

while getopts ":hf:" option; do
   case $option in
      h) # display Help
         help
         exit;;
      f) # filename to convert
         filename=${OPTARG};;
      \?) # Invalid option
         echo "Error: Invalid option"
         exit;;
   esac
done

if [ -z "${PELE_PHYSICS_HOME+xxx}" ]; then
    remove="Mechanisms"
    script_dir=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
    PELE_PHYSICS_HOME=${script_dir%%"${remove}"}
    echo "Using PELE_PHYSICS_HOME: ${PELE_PHYSICS_HOME}. Set PELE_PHYSICS_HOME you want a different one."
else
    echo "Using PELE_PHYSICS_HOME: ${PELE_PHYSICS_HOME}"
fi

CEPTR_HOME="${PELE_PHYSICS_HOME}/Support/ceptr"

case $filename in
  /*) ;; # filename is an absolute path
  *) filename="$(abspath $filename)/${filename}" ;;
esac

cd "${CEPTR_HOME}" || exit
echo "Converting ${filename}"

if command -v uv &> /dev/null
then
    uv run --locked convert -f "${filename}"
else
    echo "uv could not be found. We recommend the use of uv to ensure all necessary packages are available."
    echo "However, this script will proceed with the current python environment (and hope all the packages are available)."
    python3 -m ceptr -f "${filename}"
fi


