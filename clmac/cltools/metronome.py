"""metronome where the user inputs the bpm with a timer printed in the terminal"""
import time
from threading import Thread


def input_bpm(default: int = 130) -> int:
    bpm: str = input(f'BPM [{default}]:')
    try:
        bpm_int = int(bpm) if bpm else default
    except ValueError:
        raise ValueError('Enter a valid integer you dingus...')
    return bpm_int


def terminal_timer(refresh_period: float = 1) -> None:
    """periodically print time elapsed to the terminal"""
    start = time.time()
    while True:
        secs_elapsed = int(time.time() - start)
        print(f'time elapsed: {slibtk.human_readable_seconds(secs_elapsed)}', end='\r')
        time.sleep(refresh_period)


def get_secs_per_beat():
    user_bpm = input_bpm()
    return 60 / user_bpm


def main():
    secs_per_beat = get_secs_per_beat()
    thread_timer = Thread(target=terminal_timer, daemon=True)
    thread_timer.start()
    while True:
        time.sleep(secs_per_beat)


if __name__ == '__main__':
    main()
