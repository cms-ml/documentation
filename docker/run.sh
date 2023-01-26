#!/usr/bin/env bash

# This script builds and/or serves the documentation inside a docker container.
# Arguments:
#  1. The mode. Must be 'build', 'serve', or 'bash'. Defaults to 'build'.
#  2. The host port of the server when mode is 'serve'. Defaults to '8000'.

action() {
    local this_file="$( [ ! -z "${ZSH_VERSION}" ] && echo "${(%):-%x}" || echo "${BASH_SOURCE[0]}" )"
    local this_dir="$( cd "$( dirname "${this_file}" )" && pwd )"
    local root_dir="$( dirname "${this_dir}" )"

    # get the mode
    local mode="${1:-build}"

    # define docker args and cmd depending on the mode
    local docker_args="--rm -v \"${root_dir}\":/documentation"
    local docker_cmd=""
    if [ "${mode}" = "build" ]; then
        docker_cmd="mkdocs build --strict"
        docker_args="${docker_args} -t"
    elif [ "${mode}" = "serve" ]; then
        local host_port="${2:-8000}"
        docker_cmd="mkdocs serve --dev-addr 0.0.0.0:8000"
        docker_args="${docker_args} -t -p ${host_port}:8000"
    elif [ "${mode}" = "bash" ]; then
        docker_cmd="bash"
        docker_args="${docker_args} -ti"
    else
        2>&1 echo "unknown mode '${mode}'"
        return "1"
    fi

    # start the container
    local cmd="docker run ${docker_args} cmsml/documentation ${docker_cmd}"
    echo -e "command: ${cmd}\n"
    eval "${cmd}"
}
action "$@"
