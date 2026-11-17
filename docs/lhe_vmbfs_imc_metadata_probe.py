#!/usr/bin/env python3
"""Read only metadata for the fixed VMBFS IMC hive path.

The probe temporarily binds a GPL VMBus driver to the already-offered Initial
Machine Configuration device, negotiates the documented VMBFS protocol, and
issues GET_FILE_INFO for the fixed path /imc.hiv.  It does not issue READ_FILE,
copy file content, enumerate paths, access another VM, or use the network.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path


MODULE = "fairchem_vmbfs_info"

C_SOURCE = r"""
#include <linux/completion.h>
#include <linux/hyperv.h>
#include <linux/init.h>
#include <linux/module.h>
#include <linux/types.h>
#include <asm/unaligned.h>

#define RESPONSE_MAX 128
#define RESPONSE_TIMEOUT (2 * HZ)
#define VMBFS_VERSION_WIN10 0x00010000

struct probe_state {
    struct hv_device *hdev;
    struct completion response_ready;
    u8 response[RESPONSE_MAX];
    u32 response_len;
    bool channel_open;
};

static struct probe_state state;
static int device_bound;
static int instance_is_imc;
static int channel_opened;
static int version_request_sent;
static int version_status = -1;
static int file_info_request_sent;
static int file_status = -1;
static unsigned long long file_size_bytes;
static int probe_finished;

module_param(device_bound, int, 0444);
module_param(instance_is_imc, int, 0444);
module_param(channel_opened, int, 0444);
module_param(version_request_sent, int, 0444);
module_param(version_status, int, 0444);
module_param(file_info_request_sent, int, 0444);
module_param(file_status, int, 0444);
module_param(file_size_bytes, ullong, 0444);
module_param(probe_finished, int, 0444);

static const guid_t expected_imc_instance =
    GUID_INIT(0xc4e5e7d1, 0xd748, 0x4afc, 0x97, 0x9d,
              0x68, 0x31, 0x67, 0x91, 0x0a, 0x55);

static void vmbfs_callback(void *context)
{
    struct probe_state *s = context;
    u32 len = 0;
    u64 request_id = 0;
    int ret;

    ret = vmbus_recvpacket(s->hdev->channel, s->response,
                           sizeof(s->response), &len, &request_id);
    if (!ret && len) {
        s->response_len = len;
        complete(&s->response_ready);
    }
}

static int send_and_wait(struct probe_state *s, void *buffer, u32 length)
{
    int ret;

    reinit_completion(&s->response_ready);
    s->response_len = 0;
    ret = vmbus_sendpacket(s->hdev->channel, buffer, length, 0,
                           VM_PKT_DATA_INBAND, 0);
    if (ret)
        return ret;
    if (!wait_for_completion_timeout(&s->response_ready, RESPONSE_TIMEOUT))
        return -ETIMEDOUT;
    return 0;
}

static int parse_response(struct probe_state *s, u32 expected_type,
                          const u8 **body, u32 *body_len)
{
    u32 packet_type;
    u32 data_size;
    const u8 *payload;

    if (s->response_len < 16)
        return -EMSGSIZE;
    packet_type = get_unaligned_le32(&s->response[0]);
    data_size = get_unaligned_le32(&s->response[4]);
    if (packet_type != 1 || data_size > s->response_len - 8 || data_size < 8)
        return -EPROTO;
    payload = &s->response[8];
    if (get_unaligned_le32(&payload[0]) != expected_type)
        return -EPROTO;
    *body = &payload[8];
    *body_len = data_size - 8;
    return 0;
}

static int query_version(struct probe_state *s)
{
    struct {
        __le32 pipe_type;
        __le32 data_size;
        __le32 message_type;
        __le32 reserved;
        __le32 version;
    } __packed request = {
        .pipe_type = cpu_to_le32(1),
        .data_size = cpu_to_le32(12),
        .message_type = cpu_to_le32(1),
        .reserved = 0,
        .version = cpu_to_le32(VMBFS_VERSION_WIN10),
    };
    const u8 *body;
    u32 body_len;
    int ret;

    version_request_sent = 1;
    ret = send_and_wait(s, &request, sizeof(request));
    if (ret)
        return ret;
    ret = parse_response(s, 2, &body, &body_len);
    if (ret || body_len < 4)
        return ret ? ret : -EMSGSIZE;
    version_status = get_unaligned_le32(body);
    return version_status == 0 ? 0 : -EPROTONOSUPPORT;
}

static int query_imc_file_info(struct probe_state *s)
{
    static const u16 path[] = {'/', 'i', 'm', 'c', '.', 'h', 'i', 'v'};
    struct {
        __le32 pipe_type;
        __le32 data_size;
        __le32 message_type;
        __le32 reserved;
        __le16 path[ARRAY_SIZE(path)];
    } __packed request;
    const u8 *body;
    u32 body_len;
    unsigned int i;
    int ret;

    memset(&request, 0, sizeof(request));
    request.pipe_type = cpu_to_le32(1);
    request.data_size = cpu_to_le32(sizeof(request) - 8);
    request.message_type = cpu_to_le32(3);
    for (i = 0; i < ARRAY_SIZE(path); i++)
        request.path[i] = cpu_to_le16(path[i]);

    file_info_request_sent = 1;
    ret = send_and_wait(s, &request, sizeof(request));
    if (ret)
        return ret;
    ret = parse_response(s, 4, &body, &body_len);
    if (ret || body_len < 16)
        return ret ? ret : -EMSGSIZE;
    file_status = get_unaligned_le32(&body[0]);
    if (file_status == 0)
        file_size_bytes = get_unaligned_le64(&body[8]);
    return 0;
}

static int fairchem_vmbfs_probe(struct hv_device *dev,
                                const struct hv_vmbus_device_id *id)
{
    int ret;

    device_bound = 1;
    state.hdev = dev;
    init_completion(&state.response_ready);
    hv_set_drvdata(dev, &state);

    instance_is_imc = guid_equal(&dev->dev_instance, &expected_imc_instance);
    if (!instance_is_imc) {
        probe_finished = 1;
        return 0;
    }

    set_channel_read_mode(dev->channel, HV_CALL_DIRECT);
    ret = vmbus_open(dev->channel,
                     VMBUS_RING_SIZE(3 * HV_HYP_PAGE_SIZE),
                     VMBUS_RING_SIZE(3 * HV_HYP_PAGE_SIZE),
                     NULL, 0, vmbfs_callback, &state);
    if (ret) {
        probe_finished = 1;
        return 0;
    }
    state.channel_open = true;
    channel_opened = 1;

    ret = query_version(&state);
    if (!ret)
        query_imc_file_info(&state);
    probe_finished = 1;
    return 0;
}

static void fairchem_vmbfs_remove(struct hv_device *dev)
{
    struct probe_state *s = hv_get_drvdata(dev);

    if (s && s->channel_open) {
        vmbus_close(dev->channel);
        s->channel_open = false;
    }
}

static const struct hv_vmbus_device_id fairchem_vmbfs_id_table[] = {
    { HV_IMC_GUID },
    { },
};
MODULE_DEVICE_TABLE(vmbus, fairchem_vmbfs_id_table);

static struct hv_driver fairchem_vmbfs_driver = {
    .name = "fairchem_vmbfs_info",
    .id_table = fairchem_vmbfs_id_table,
    .probe = fairchem_vmbfs_probe,
    .remove = fairchem_vmbfs_remove,
};

static int __init fairchem_vmbfs_init(void)
{
    return vmbus_driver_register(&fairchem_vmbfs_driver);
}

static void __exit fairchem_vmbfs_exit(void)
{
    vmbus_driver_unregister(&fairchem_vmbfs_driver);
}

module_init(fairchem_vmbfs_init);
module_exit(fairchem_vmbfs_exit);
MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("Meta LHE bounded VMBFS IMC metadata probe");
"""


def run(argv: list[str], timeout: int = 120) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            argv,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError:
        return subprocess.CompletedProcess(argv, 127, "", "command-not-found")
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(argv, 124, "", "timeout")


def read_parameter(name: str) -> int | None:
    try:
        return int((Path("/sys/module") / MODULE / "parameters" / name).read_text().strip())
    except (OSError, ValueError):
        return None


def load_error_class(stderr: str) -> str:
    value = stderr.lower()
    if "unknown symbol" in value:
        return "unknown-symbol"
    if "invalid module format" in value:
        return "module-format"
    if "permission denied" in value or "not permitted" in value:
        return "permission-policy"
    if "device or resource busy" in value:
        return "device-busy"
    return "other" if stderr.strip() else "none"


def status_class(value: int | None) -> str:
    return {
        0: "success",
        1: "not-found",
        2: "end-of-file",
        3: "host-error",
        -1: "no-response",
    }.get(value, "other")


def main() -> int:
    release = os.uname().release
    build = Path("/lib/modules") / release / "build"
    result: dict[str, object] = {
        "schema": 1,
        "euid_is_root": os.geteuid() == 0,
        "hyperv_guest": Path("/sys/bus/vmbus").exists(),
        "build_tree_present": build.is_dir(),
        "compiler_present": shutil.which("cc") is not None,
        "make_present": shutil.which("make") is not None,
        "insmod_present": shutil.which("insmod") is not None,
        "rmmod_present": shutil.which("rmmod") is not None,
        "fixed_path_class": "vmbfs-imc-hive",
        "path_enumeration_performed": False,
        "read_file_request_sent": False,
        "host_file_content_read": False,
        "network_requested": False,
        "module_build_attempted": False,
        "module_built": False,
        "module_build_diagnostic_tail": "",
        "module_load_attempted": False,
        "module_loaded": False,
        "module_load_error_class": "not-attempted",
        "module_removed": False,
    }

    can_build = build.is_dir() and shutil.which("make") and shutil.which("cc")
    can_load = shutil.which("insmod") and shutil.which("rmmod")
    if can_build:
        result["module_build_attempted"] = True
        with tempfile.TemporaryDirectory(prefix="fairchem-lhe-vmbfs-") as temporary:
            directory = Path(temporary)
            (directory / f"{MODULE}.c").write_text(C_SOURCE)
            (directory / "Makefile").write_text(f"obj-m += {MODULE}.o\n")
            built = run(["make", "-s", "-C", str(build), f"M={directory}", "modules"])
            module_path = directory / f"{MODULE}.ko"
            result["module_built"] = built.returncode == 0 and module_path.is_file()
            if not result["module_built"]:
                diagnostic = (built.stdout + "\n" + built.stderr).replace(
                    str(directory), "<temporary-module-directory>"
                )
                result["module_build_diagnostic_tail"] = diagnostic[-1600:]

            if result["module_built"] and can_load:
                result["module_load_attempted"] = True
                loaded = run(["insmod", str(module_path)], timeout=30)
                module_loaded = loaded.returncode == 0 and Path(f"/sys/module/{MODULE}").exists()
                result["module_loaded"] = module_loaded
                result["module_load_error_class"] = load_error_class(loaded.stderr)
                if module_loaded:
                    for _ in range(50):
                        if read_parameter("probe_finished") == 1:
                            break
                        time.sleep(0.1)
                    result.update(
                        {
                            "device_bound": read_parameter("device_bound") == 1,
                            "instance_is_imc": read_parameter("instance_is_imc") == 1,
                            "channel_opened": read_parameter("channel_opened") == 1,
                            "version_request_sent": read_parameter("version_request_sent") == 1,
                            "version_status_class": status_class(read_parameter("version_status")),
                            "file_info_request_sent": read_parameter("file_info_request_sent") == 1,
                            "file_status_class": status_class(read_parameter("file_status")),
                            "file_size_bytes": read_parameter("file_size_bytes"),
                            "probe_finished": read_parameter("probe_finished") == 1,
                        }
                    )
                    removed = run(["rmmod", MODULE], timeout=30)
                    result["module_removed"] = (
                        removed.returncode == 0 and not Path(f"/sys/module/{MODULE}").exists()
                    )

    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
