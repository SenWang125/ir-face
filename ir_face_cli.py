#!/usr/bin/env python3
"""ir-face — manage IR face authentication"""

import sys
import os
import time
import configparser
import numpy as np

MODEL_DIR   = "/etc/ir-face/models"
CONFIG_PATH = "/etc/ir-face/config.ini"


def get_user():
    return os.environ.get("SUDO_USER") or os.environ.get("USER") or os.getlogin()


def cmd_list(args):
    username = args[0] if args else get_user()
    path = os.path.join(MODEL_DIR, f"{username}.npy")
    if not os.path.exists(path):
        print(f"No enrolled face for '{username}'")
        return 1
    data = np.load(path, allow_pickle=True).item()
    models = data.get("models", [])
    print(f"Enrolled models for '{username}':")
    for m in models:
        ts    = time.strftime("%Y-%m-%d %H:%M", time.localtime(m.get("created", 0)))
        count = m.get("frame_count", len(m.get("embeddings", [])))
        det   = m.get("det_pack", "?")
        rec   = m.get("rec_pack", "?")
        print(f"  {m['label']:20s}  {count:3d} frames  det={det} rec={rec}  enrolled {ts}")
    return 0


def cmd_add(args):
    label = args[0] if args else "Default"
    user  = get_user()
    os.execv("/usr/local/bin/ir-enroll", ["/usr/local/bin/ir-enroll", user, label])


def cmd_remove(args):
    if not args:
        print("Usage: ir-face remove <label> [username]")
        return 1
    label    = args[0]
    username = args[1] if len(args) > 1 else get_user()
    path = os.path.join(MODEL_DIR, f"{username}.npy")
    if not os.path.exists(path):
        print(f"No model for '{username}'")
        return 1
    data   = np.load(path, allow_pickle=True).item()
    models = data.get("models", [])
    before = len(models)
    models = [m for m in models if m["label"] != label]
    if len(models) == before:
        print(f"No model labeled '{label}' for '{username}'")
        return 1
    if not models:
        os.remove(path)
        print(f"Removed '{label}' — no models remain, deleted {path}")
    else:
        data["models"]     = models
        data["embeddings"] = np.vstack([m["embeddings"] for m in models])
        np.save(path, data, allow_pickle=True)
        print(f"Removed '{label}' for '{username}'. Remaining: {[m['label'] for m in models]}")
    return 0


def cmd_test(args):
    user = args[0] if args else get_user()
    os.execv("/usr/local/bin/ir-compare", ["/usr/local/bin/ir-compare", user, "--verbose"])


def cmd_config(args):
    config = configparser.ConfigParser()
    config.read(CONFIG_PATH)
    if not args:
        for section in config.sections():
            print(f"[{section}]")
            for k, v in config.items(section):
                print(f"  {k} = {v}")
        return 0
    key = args[0]
    if len(args) == 1:
        for section in config.sections():
            if config.has_option(section, key):
                print(config.get(section, key))
                return 0
        print(f"Key '{key}' not found in {CONFIG_PATH}")
        return 1
    value = args[1]
    for section in config.sections():
        if config.has_option(section, key):
            config.set(section, key, value)
            import io, subprocess
            buf = io.StringIO()
            config.write(buf)
            r = subprocess.run(
                ["sudo", "tee", CONFIG_PATH],
                input=buf.getvalue().encode(),
                capture_output=True,
            )
            if r.returncode != 0:
                print(f"Write failed (try: sudo ir-face config {key} {value})")
                return 1
            print(f"{key} = {value}")
            return 0
    print(f"Key '{key}' not found in {CONFIG_PATH}")
    return 1


COMMANDS = {
    "list":   (cmd_list,   "list   [user]             show enrolled models"),
    "add":    (cmd_add,    "add    [label]            enroll face (default: Default)"),
    "remove": (cmd_remove, "remove <label> [user]     delete a model label"),
    "test":   (cmd_test,   "test   [user]             run recognition with verbose output"),
    "config": (cmd_config, "config [key [value]]      view or edit config.ini"),
}


def usage():
    print("Usage: ir-face <command> [args]")
    print()
    print("Commands:")
    for _, (_, desc) in COMMANDS.items():
        print(f"  {desc}")
    print()
    print("Notes:")
    print("  add / remove / config set — require sudo (write to /etc/ir-face/)")
    print("  test — runs with verbose recognition output")


def main():
    argv = sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help", "help"):
        usage()
        return 0
    cmd = argv[0]
    if cmd not in COMMANDS:
        print(f"Unknown command: {cmd}")
        usage()
        return 1
    return COMMANDS[cmd][0](argv[1:]) or 0


if __name__ == "__main__":
    sys.exit(main())
