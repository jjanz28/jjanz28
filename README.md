# Sudo Convenience Aliases
This machine is configured with Bash aliases that automatically prefix common admin commands with `sudo`.

## Alias file
Aliases are defined in:

- `~/.bash_aliases`

They are loaded from:

- `~/.bashrc`

## Configured aliases
- `apt='sudo apt'`
- `apt-get='sudo apt-get'`
- `apt-cache='sudo apt-cache'`
- `dpkg='sudo dpkg'`
- `systemctl='sudo systemctl'`
- `service='sudo service'`
- `journalctl='sudo journalctl'`
- `ufw='sudo ufw'`
- `mount='sudo mount'`
- `umount='sudo umount'`
- `snap='sudo snap'`

## Activate in current shell
Run:

```bash
source ~/.bashrc
```

## Verify aliases
Run:

```bash
alias apt
alias systemctl
```

## Safely add or remove aliases
### Add an alias
1. Edit `~/.bash_aliases`.
2. Add one alias per line, for example:

```bash
alias ll='ls -alF'
```

3. Reload shell config:

```bash
source ~/.bashrc
```

4. Verify:

```bash
alias ll
```

### Remove an alias
- Permanent removal:
  1. Delete the alias line from `~/.bash_aliases`.
  2. Run `source ~/.bashrc`.
- Temporary removal (current shell only):

```bash
unalias ll
```

### Safety tips
- Avoid aliasing critical commands to risky behavior (for example `rm` with destructive flags).
- After changes, always test with `alias <name>` before relying on the alias.
- If an alias interferes with a command, run the real command with a leading backslash, e.g. `\ls`.
