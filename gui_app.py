#!/usr/bin/env python3
"""Desktop GUI launcher for local Stable Diffusion generation."""

from __future__ import annotations

import subprocess
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk

from presets import DEFAULT_PRESETS_PATH, get_preset, list_presets, save_preset

PROJECT_DIR = Path(__file__).resolve().parent
VENV_PYTHON = PROJECT_DIR / ".venv" / "bin" / "python"
GENERATE_SCRIPT = PROJECT_DIR / "generate.py"
OUTPUT_DIR = PROJECT_DIR / "outputs"
DEFAULT_MODEL = "runwayml/stable-diffusion-v1-5"
DEFAULT_STEPS = 40
DEFAULT_GUIDANCE_SCALE = 9.0
DEFAULT_SCHEDULER = "dpmpp_2m"
DEFAULT_PRESET_NAME = "fantasy-vivid"
DEFAULT_NEGATIVE_PROMPT = (
    "blurry, low quality, dull colors, muted colors, low contrast, washed out, "
    "bad anatomy, deformed, watermark, text, logo"
)
DEFAULT_PROMPT_TEMPLATE = (
    "epic fantasy landscape, magical atmosphere, intricate details, "
    "vibrant colors, dramatic lighting, volumetric light, cinematic composition"
)


class StableDiffusionLauncher:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Stable Diffusion Launcher")
        self.root.geometry("760x520")

        self.negative_prompt_var = tk.StringVar(value=DEFAULT_NEGATIVE_PROMPT)
        self.model_var = tk.StringVar(value=DEFAULT_MODEL)
        self.output_var = tk.StringVar(value=str(OUTPUT_DIR / "gui_output.png"))
        self.steps_var = tk.IntVar(value=DEFAULT_STEPS)
        self.scheduler_var = tk.StringVar(value=DEFAULT_SCHEDULER)
        self.guidance_scale_var = tk.DoubleVar(value=DEFAULT_GUIDANCE_SCALE)
        self.low_memory_var = tk.BooleanVar(value=False)
        self.seed_var = tk.StringVar()
        self.preset_var = tk.StringVar(value=DEFAULT_PRESET_NAME)
        self.presets_file = DEFAULT_PRESETS_PATH
        self.available_presets: list[str] = []

        self.prompt_text: tk.Text
        self.log_widget: tk.Text
        self.generate_button: ttk.Button
        self.preset_combo: ttk.Combobox
        self.status_var = tk.StringVar(value="Ready.")

        self._build_ui()
        self._refresh_presets(selected=DEFAULT_PRESET_NAME)
        if self.preset_var.get():
            self._apply_selected_preset()

    def _build_ui(self) -> None:
        frame = ttk.Frame(self.root, padding=12)
        frame.pack(fill="both", expand=True)

        ttk.Label(frame, text="Prompt").grid(row=0, column=0, sticky="w")
        self.prompt_text = tk.Text(frame, height=5, wrap="word")
        self.prompt_text.grid(row=1, column=0, columnspan=4, sticky="ew", pady=(4, 10))
        self.prompt_text.insert("1.0", DEFAULT_PROMPT_TEMPLATE)

        ttk.Label(frame, text="Negative Prompt").grid(row=2, column=0, sticky="w")
        ttk.Entry(frame, textvariable=self.negative_prompt_var).grid(
            row=3, column=0, columnspan=4, sticky="ew", pady=(4, 10)
        )

        ttk.Label(frame, text="Model").grid(row=4, column=0, sticky="w")
        ttk.Entry(frame, textvariable=self.model_var).grid(
            row=5, column=0, columnspan=4, sticky="ew", pady=(4, 10)
        )

        ttk.Label(frame, text="Output").grid(row=6, column=0, sticky="w")
        ttk.Entry(frame, textvariable=self.output_var).grid(
            row=7, column=0, columnspan=3, sticky="ew", pady=(4, 10)
        )
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
            values=("default", "euler", "euler_a", "ddim", "dpmpp_2m"),
            state="readonly",
            width=12,
        ).grid(row=9, column=1, sticky="w", pady=(4, 10))

        ttk.Label(frame, text="Guidance").grid(row=8, column=2, sticky="w")
        ttk.Spinbox(
            frame,
            from_=1.0,
            to=20.0,
            increment=0.5,
            textvariable=self.guidance_scale_var,
            width=8,
        ).grid(row=9, column=2, sticky="w", pady=(4, 10))

        ttk.Label(frame, text="Seed (optional)").grid(row=8, column=3, sticky="w")
        ttk.Entry(frame, textvariable=self.seed_var, width=14).grid(
            row=9, column=3, sticky="w", pady=(4, 10)
        )

        ttk.Checkbutton(frame, text="Low memory mode", variable=self.low_memory_var).grid(
            row=10, column=0, sticky="w", pady=(4, 10)
        )
        ttk.Label(frame, text="Preset").grid(row=10, column=1, sticky="e", pady=(4, 10))
        self.preset_combo = ttk.Combobox(
            frame,
            textvariable=self.preset_var,
            state="readonly",
            width=22,
        )
        self.preset_combo.grid(row=10, column=2, sticky="ew", pady=(4, 10))
        preset_buttons = ttk.Frame(frame)
        preset_buttons.grid(row=10, column=3, sticky="w", pady=(4, 10))
        ttk.Button(preset_buttons, text="Apply", command=self._apply_selected_preset).pack(
            side="left"
        )
        ttk.Button(
            preset_buttons, text="Save", command=self._save_current_as_preset
        ).pack(side="left", padx=(6, 0))

        buttons = ttk.Frame(frame)
        buttons.grid(row=11, column=0, columnspan=4, sticky="ew", pady=(2, 10))
        self.generate_button = ttk.Button(buttons, text="Generate", command=self._generate)
        self.generate_button.pack(side="left")
        ttk.Button(buttons, text="Open Outputs Folder", command=self._open_outputs).pack(
            side="left", padx=(8, 0)
        )

        ttk.Label(frame, textvariable=self.status_var).grid(
            row=12, column=0, columnspan=4, sticky="w"
        )
        ttk.Label(frame, text="Logs").grid(row=13, column=0, sticky="w", pady=(10, 4))
        self.log_widget = tk.Text(frame, height=10, wrap="word", state="disabled")
        self.log_widget.grid(row=14, column=0, columnspan=4, sticky="nsew")

        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)
        frame.columnconfigure(2, weight=1)
        frame.columnconfigure(3, weight=0)
        frame.rowconfigure(14, weight=1)

    def _refresh_presets(self, selected: str | None = None) -> None:
        try:
            self.available_presets = list_presets(self.presets_file)
        except ValueError as err:
            self.available_presets = []
            messagebox.showerror("Preset error", str(err))

        self.preset_combo.configure(values=self.available_presets)
        current_or_selected = (selected or self.preset_var.get()).strip()
        if current_or_selected and current_or_selected in self.available_presets:
            self.preset_var.set(current_or_selected)
            return
        if self.available_presets:
            self.preset_var.set(self.available_presets[0])
            return
        self.preset_var.set("")

    def _apply_preset_values(self, values: dict) -> None:
        if "model" in values:
            self.model_var.set(values["model"])
        if "negative_prompt" in values:
            self.negative_prompt_var.set(values["negative_prompt"] or "")
        if "scheduler" in values:
            self.scheduler_var.set(values["scheduler"])
        if "steps" in values:
            self.steps_var.set(values["steps"])
        if "guidance_scale" in values:
            self.guidance_scale_var.set(values["guidance_scale"])
        if "low_memory" in values:
            self.low_memory_var.set(values["low_memory"])
        if "seed" in values:
            self.seed_var.set(str(values["seed"]))

    def _apply_selected_preset(self) -> None:
        preset_name = self.preset_var.get().strip()
        if not preset_name:
            self.status_var.set("No preset selected.")
            return
        try:
            preset = get_preset(preset_name, self.presets_file)
        except (ValueError, KeyError) as err:
            messagebox.showerror("Preset error", str(err))
            return
        self._apply_preset_values(preset)
        self.status_var.set(f"Applied preset: {preset_name}")

    def _collect_current_preset_values(self) -> dict:
        negative_prompt = self.negative_prompt_var.get().strip()
        values = {
            "model": self.model_var.get().strip() or DEFAULT_MODEL,
            "scheduler": self.scheduler_var.get(),
            "steps": self.steps_var.get(),
            "guidance_scale": float(self.guidance_scale_var.get()),
            "negative_prompt": negative_prompt or None,
            "low_memory": bool(self.low_memory_var.get()),
        }
        seed_text = self.seed_var.get().strip()
        if seed_text and seed_text.isdigit():
            values["seed"] = int(seed_text)
        return values

    def _save_current_as_preset(self) -> None:
        initial_name = self.preset_var.get().strip() or "new-preset"
        preset_name = simpledialog.askstring(
            "Save preset",
            "Preset name:",
            initialvalue=initial_name,
            parent=self.root,
        )
        if not preset_name:
            return

        try:
            save_preset(
                name=preset_name,
                values=self._collect_current_preset_values(),
                presets_file=self.presets_file,
            )
        except ValueError as err:
            messagebox.showerror("Preset error", str(err))
            return

        cleaned_name = preset_name.strip()
        self._refresh_presets(selected=cleaned_name)
        self.status_var.set(f"Saved preset: {cleaned_name}")

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
        guidance_scale = self.guidance_scale_var.get()
        if guidance_scale <= 0:
            raise ValueError("Guidance must be > 0.")

        command = [
            str(VENV_PYTHON),
            str(GENERATE_SCRIPT),
            "--prompt",
            prompt,
        ]
        preset_name = self.preset_var.get().strip()
        if preset_name:
            command.extend(["--preset", preset_name, "--presets-file", str(self.presets_file)])

        command.extend(
            [
                "--model",
                self.model_var.get().strip() or DEFAULT_MODEL,
                "--output",
                str(output_path),
                "--steps",
                str(self.steps_var.get()),
                "--guidance-scale",
                str(guidance_scale),
                "--scheduler",
                self.scheduler_var.get(),
            ]
        )

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