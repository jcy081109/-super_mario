from __future__ import annotations

import argparse
import time
from typing import Iterable

import gym_super_mario_bros
import pyglet
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py._image_viewer import ImageViewer
from nes_py.wrappers import JoypadSpace


DEFAULT_ENV_ID = "SuperMarioBros-1-1-v0"
KEY = pyglet.window.key
RELEVANT_KEYS = [
    KEY.LEFT,
    KEY.RIGHT,
    KEY.UP,
    KEY.DOWN,
    KEY.A,
    KEY.D,
    KEY.W,
    KEY.S,
    ord(" "),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play Super Mario Bros with keyboard.")
    parser.add_argument("--env-id", default=DEFAULT_ENV_ID)
    parser.add_argument("--fps", type=int, default=60)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    patch_nes_py_numpy_scalar_math()

    env = gym_super_mario_bros.make(args.env_id)
    if env.__class__.__name__ == "TimeLimit":
        env = env.env
    env = JoypadSpace(env, SIMPLE_MOVEMENT)

    raw_env = env.unwrapped
    raw_env.viewer = ImageViewer(
        caption=f"{args.env_id} - keyboard play",
        height=240,
        width=256,
        monitor_keyboard=True,
        relevant_keys=RELEVANT_KEYS,
    )

    env.reset()
    env.render(mode="human")
    should_close = False

    @raw_env.viewer._window.event
    def on_close() -> None:
        nonlocal should_close
        should_close = True
        raw_env.viewer._window.close()

    frame_delay = 1.0 / max(1, args.fps)
    print("Controls: Left/Right or A/D, Space/W/Up to jump, Down/S to run, Esc to quit.")

    try:
        while not should_close and raw_env.viewer is not None and raw_env.viewer.is_open:
            started = time.perf_counter()
            action = action_from_keys(raw_env.viewer.pressed_keys)
            _, _, done, info = env.step(action)

            if should_close or raw_env.viewer.is_escape_pressed:
                break

            env.render(mode="human")

            if should_close or raw_env.viewer.is_escape_pressed:
                break
            if done:
                print(
                    "Episode ended: "
                    f"x_pos={info.get('x_pos', '?')} "
                    f"status={info.get('status', '?')} "
                    f"flag_get={info.get('flag_get', False)}"
                )
                env.reset()

            elapsed = time.perf_counter() - started
            if elapsed < frame_delay:
                time.sleep(frame_delay - elapsed)
    finally:
        try:
            env.close()
        except ValueError:
            pass


def action_from_keys(pressed_keys: Iterable[int]) -> int:
    keys = set(pressed_keys)
    moving_right = KEY.RIGHT in keys or KEY.D in keys
    moving_left = KEY.LEFT in keys or KEY.A in keys
    jumping = ord(" ") in keys or KEY.UP in keys or KEY.W in keys
    running = KEY.DOWN in keys or KEY.S in keys

    if moving_right and jumping and running:
        return 4
    if moving_right and running:
        return 3
    if moving_right and jumping:
        return 2
    if moving_right:
        return 1
    if jumping:
        return 5
    if moving_left:
        return 6
    return 0


def patch_nes_py_numpy_scalar_math() -> None:
    try:
        from nes_py._rom import ROM
    except ImportError:
        return

    if getattr(ROM, "_mario_play_patch", False):
        return

    ROM.prg_rom_size = property(lambda self: 16 * int(self.header[4]))
    ROM.chr_rom_size = property(lambda self: 8 * int(self.header[5]))

    def prg_ram_size(self):
        size = int(self.header[8])
        if size == 0:
            size = 1
        return 8 * size

    ROM.prg_ram_size = property(prg_ram_size)
    ROM._mario_play_patch = True


if __name__ == "__main__":
    main()
