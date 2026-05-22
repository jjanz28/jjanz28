# AMD CPU Monitoring Configuration (E-SMI + lm-sensors + zenpower)
This system is configured to use `lm-sensors` with `zenpower` for Ryzen CPU telemetry, with E-SMI installed for compatible metrics.

## Installed components
- E-SMI binary and library (user-local install):
  - `~/.local/bin/e_smi_tool`
  - `~/.local/lib/libe_smi64.so`
- `lm-sensors` package
- `zenpower` DKMS module (installed from maintained `zenpower5` source)

## Shell environment for E-SMI
The following are added in `~/.bashrc` with duplicate-safe guards:
- `PATH` includes `~/.local/bin`
- `LD_LIBRARY_PATH` includes `~/.local/lib`

Apply immediately in current shell:
```bash
source ~/.bashrc
```

## Persistent zenpower setup
Because `zenpower` and `k10temp` can conflict, the system is configured to prefer `zenpower`.

### 1) Blacklist k10temp
File: `/etc/modprobe.d/zenpower.conf`
```conf
# Use zenpower instead of k10temp for AMD CPU telemetry
blacklist k10temp
```

### 2) Auto-load zenpower on boot
File: `/etc/modules-load.d/zenpower.conf`
```conf
zenpower
```

## Current expected runtime state
- `zenpower` module loaded
- `k10temp` not loaded
- `sensors` output includes `zenpower-pci-00c3` with:
  - `Tdie`, `Tctl`
  - `SVI2_Core`, `SVI2_SoC`
  - `SVI2_P_Core`, `SVI2_P_SoC`
  - `SVI2_C_Core`, `SVI2_C_SoC`

## Verification after reboot
Run:
```bash
lsmod | grep -E 'zenpower|k10temp' || true
echo '---'
sudo cat /etc/modprobe.d/zenpower.conf
echo '---'
sudo cat /etc/modules-load.d/zenpower.conf
echo '---'
sensors
```

Expected:
- `lsmod` shows `zenpower`
- `k10temp` is absent
- config files match the snippets above
- `sensors` shows `zenpower-pci-00c3` metrics

## Notes
- On this hardware/firmware path, HSMP-backed E-SMI features are unavailable; E-SMI still installs and runs but advanced HSMP commands may report unsupported/disabled.
