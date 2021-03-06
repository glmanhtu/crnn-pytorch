#!/bin/bash

MACHINE="deep3"
USER="mvu"
JUMP_HOST='ssh.labri.fr'
SYNC_DIR='/data2/mvu/phd_experiments/*'
TO_DIR='.'
POSITIONAL=()
while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    -m|--machine)
    MACHINE="$2"
    shift # past argument
    shift # past value
    ;;
    -u|--user)
    USER="$2"
    shift # past argument
    shift # past value
    ;;
    -j|--jump-host)
    JUMP_HOST="$2"
    shift # past argument
    shift # past value
    ;;
    -s|--sync-dir)
    SYNC_DIR="$2"
    shift # past argument
    shift # past value
    ;;
    -t|--to-dir)
    TO_DIR="$2"
    shift # past argument
    shift # past value
    ;;
    *)    # unknown option
    POSITIONAL+=("$1") # save it in an array for later
    shift # past argument
    ;;
esac
done
set -- "${POSITIONAL[@]}" # restore positional parameters

rsync -Pva -e "ssh -J ${USER}@${JUMP_HOST}" --exclude venv --exclude resources "${USER}@${MACHINE}:${SYNC_DIR}" "${TO_DIR}"
