#!/usr/bin/env python3
"""Desktop GUI launcher for local Stable Diffusion generation."""

from __future__ import annotations

import subprocess
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

PROJECT_DIR = Path(__file__).resolve().parent
VENV_PYTHON = PROJECT_DIR / ".venv" / "bin" / "python"
GENERATE_SCRIPT = PROJECT_DIR / "generate.py"
OUTPUT_DIR = PROJECT_DIR / "outputs"
DEFAULT_MODEL = "runwayml/stable-diffusion-v1-5"


class StableDiffusionLauncher:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Stable Diffusion Launcher")
        self.root.geometry("760x520")

        self.negative_prompt_var = tk.StringVar()
        self.model_var = tk.StringVar(value=DEFAULT_MODEL)
        self.output_var = tk.StringVar(value=str(OUTPUT_DIR / "gui_output.png"))
        self.steps_var = tk.IntVar(value=25)
        self.scheduler_var = tk.StringVar(value="default")
        self.low_memory_var = tk.BooleanVar(value=True)
        self.seed_var = tk.StringVar()
        self.prompt_text: tk.Text

        self.generate_button: ttk.Button
        self.status_var = tk.StringVar(value="Ready.")
        self.log_widget: tk.Text

        self._build_ui()

    def _build_ui(self) -> None:
        frame = ttk.Frame(self.root, padding=12)
        frame.pack(fill="both", expand=True)

        ttk.Label(frame, text="Prompt").grid(row=0, column=0, sticky="w")
        self.prompt_text = tk.Text(frame, height=5, wrap="word")
        self.prompt_text.grid(row=1, column=0, columnspan=4, sticky="ew", pady=(4, 10))

        ttk.Label(frame, text="Negative Prompt").grid(row=2, column=0, sticky="w")
        ttk.Entry(frame, textvariable=self.negative_prompt_var).grid(
            row=3, column=0, columnspan=4, sticky="ew", pady=(4, 10)
        )

        ttk.Label(frame, text="Model").grid(row=4, column=0, sticky="w")
        ttk.Entry(frame, textvariable=self.model_var).grid(
            row=5, column=0, columnspan=4, sticky="ew", pady=(4, 10)
        )

        ttk.Label(frame, text="Output").grid(row=6, column=0, sticky="w")
        output_entry = ttk.Entry(frame, textvariable=self.output_var)
        output_entry.grid(row=7, column=0, columnspan=3, sticky="ew", pady=(4, 10))
        ttk.Button(frame, text="Browse", command=self._choose_output).grid(
            row=7, column=3, sticky="ew", padx=(8, 0), pady=(4, 10)
        )

        ttk.Label(frame, text="Steps").grid(row=8, column=0, sticky="w")
        ttk.Spinbox(frame, from_=1, to=150, textvariable=self.steps_var, width=8).grid(
            row=9, column=0, sticky="w", pady=(4, 10)
        )

        ttk.Label(frame, text="Scheduler").grid(row=8, column=1, sticky="w")
        ttk.Combobox(
            frame,
            textvariable=self.scheduler_var,
            values=("default", "euler", "euler_a", "ddim", "dpm"),
            state="readonly",
            width=12,
        ).grid(row=9, column=1, sticky="w", pady=(4, 10))

        ttk.Label(frame, text="Seed (optional)").grid(row=8, column=2, sticky="w")
        ttk.Entry(frame, textvariable=self.seed_var, width=14).grid(
            row=9, column=2, sticky="w", pady=(4, 10)
        )

        ttk.Checkbutton(frame, text="Low memory mode", variable=self.low_memory_var).grid(
            row=9, column=3, sticky="w", pady=(4, 10)
        )

        buttons = ttk.Frame(frame)
        buttons.grid(row=10, column=0, columnspan=4, sticky="ew", pady=(2, 10))
        self.generate_button = ttk.Button(buttons, text="Generate", command=self._generate)
        self.generate_button.pack(side="left")
        ttk.Button(buttons, text="Open Outputs Folder", command=self._open_outputs).pack(
            side="left", padx=(8, 0)
        )

        ttk.Label(frame, textvariable=self.status_var).grid(
            row=11, column=0, columnspan=4, sticky="w"
        )
        ttk.Label(frame, text="Logs").grid(row=12, column=0, sticky="w", pady=(10, 4))
        self.log_widget = tk.Text(frame, height=10, wrap="word", state="disabled")
        self.log_widget.grid(row=13, column=0, columnspan=4, sticky="nsew")

        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)
        frame.columnconfigure(2, weight=1)
        frame.columnconfigure(3, weight=0)
        frame.rowconfigure(13, weight=1)

    def _choose_output(self) -> None:
        selected = filedialog.asksaveasfilename(
            title="Choose output image path",
            initialdir=str(OUTPUT_DIR),
            initialfile="gui_output.png",
            defaultextension=".png",
            filetypes=[("PNG image", "*.png"), ("All files", "*.*")],
        )
        if selected:
            self.output_var.set(selected)

    def _append_log(self, text: str) -> None:
        self.log_widget.configure(state="normal")
        self.log_widget.insert("end", text + "\n")
        self.log_widget.see("end")
        self.log_widget.configure(state="disabled")

    def _open_outputs(self) -> None:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        subprocess.run(["xdg-open", str(OUTPUT_DIR)], check=False)

    def _build_command(self) -> list[str]:
        prompt = self.prompt_text.get("1.0", "end").strip()
        if not prompt:
            raise ValueError("Prompt is required.")

        output_path = Path(self.output_var.get()).expanduser()
        if output_path.suffix.lower() != ".png":
            raise ValueError("Output file must end with .png")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        command = [
            str(VENV_PYTHON),
            str(GENERATE_SCRIPT),
            "--prompt",
            prompt,
            "--model",
            self.model_var.get().strip() or DEFAULT_MODEL,
            "--output",
            str(output_path),
            "--steps",
            str(self.steps_var.get()),
            "--scheduler",
            self.scheduler_var.get(),
        ]

        negative_prompt = self.negative_prompt_var.get().strip()
        if negative_prompt:
            command.extend(["--negative-prompt", negative_prompt])

        if self.low_memory_var.get():
            command.append("--low-memory")

        seed = self.seed_var.get().strip()
        if seed:
            if not seed.isdigit():
                raise ValueError("Seed must be an integer.")
            command.extend(["--seed", seed])

        return command

    def _generate(self) -> None:
        if not VENV_PYTHON.exists():
            messagebox.showerror("Missing venv", f"Python not found at {VENV_PYTHON}")
            return
        if not GENERATE_SCRIPT.exists():
            messagebox.showerror("Missing script", f"Could not find {GENERATE_SCRIPT}")
            return

        try:
            command = self._build_command()
        except ValueError as err:
            messagebox.showerror("Invalid input", str(err))
            return

        self.generate_button.configure(state="disabled")
        self.status_var.set("Generating image...")
        self._append_log("$ " + " ".join(command))

        thread = threading.Thread(target=self._run_generation, args=(command,), daemon=True)
        thread.start()

    def _run_generation(self, command: list[str]) -> None:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            cwd=str(PROJECT_DIR),
            check=False,
        )
        self.root.after(0, self._finish_generation, result.returncode, result.stdout, result.stderr)

    def _finish_generation(self, code: int, stdout: str, stderr: str) -> None:
        if stdout.strip():
            self._append_log(stdout.strip())
        if stderr.strip():
            self._append_log(stderr.strip())

        if code == 0:
            self.status_var.set("Generation complete.")
            messagebox.showinfo("Success", "Image generation finished successfully.")
        else:
            self.status_var.set("Generation failed.")
            messagebox.showerror("Generation failed", f"Command exited with code {code}.")

        self.generate_button.configure(state="normal")


def main() -> None:
    root = tk.Tk()
    app = StableDiffusionLauncher(root)
    _ = app
    root.mainloop()


if __name__ == "__main__":
    main()
