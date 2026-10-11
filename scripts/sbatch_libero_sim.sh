#!/usr/bin/env bash
# Submit a LIBERO/LIBERO-Plus simulator job while excluding nodes with
# confirmed rendering/EGL failures. Any caller-provided --exclude list is
# merged with config/libero_sim_node_blacklist.txt.

set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
blacklist_file=${LIBERO_SIM_NODE_BLACKLIST:-${repo_root}/config/libero_sim_node_blacklist.txt}

if [[ ! -f "${blacklist_file}" ]]; then
    echo "Missing LIBERO simulator blacklist: ${blacklist_file}" >&2
    exit 2
fi

declare -a passthrough=()
declare -a excluded=()
while IFS= read -r node; do
    node=${node%%#*}
    node=${node//[[:space:]]/}
    [[ -n "${node}" ]] && excluded+=("${node}")
done < "${blacklist_file}"

while (($#)); do
    case "$1" in
        --exclude=*)
            IFS=',' read -r -a supplied <<< "${1#--exclude=}"
            excluded+=("${supplied[@]}")
            shift
            ;;
        --exclude)
            (($# >= 2)) || { echo "--exclude requires a value" >&2; exit 2; }
            IFS=',' read -r -a supplied <<< "$2"
            excluded+=("${supplied[@]}")
            shift 2
            ;;
        *)
            passthrough+=("$1")
            shift
            ;;
    esac
done

exclude_csv=$(
    printf '%s\n' "${excluded[@]}" |
        sed '/^[[:space:]]*$/d' |
        sort -u |
        paste -sd, -
)

echo "LIBERO simulator node blacklist: ${exclude_csv}" >&2
exec sbatch --exclude="${exclude_csv}" "${passthrough[@]}"
