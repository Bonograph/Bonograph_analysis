"""Which rhythms are in a trace, and take some of them out.

X is time (or position) in equal steps; each data set is a trace.

  see the frequencies   the Fourier transform: how strong each frequency is
  filter                keep the slow part, the fast part, a band, or all but a band

Frequency is in cycles per unit of X: with X in seconds it is hertz.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal

from ..backend_types import AnalysisError, AnalysisResult, ChartSeries, ChartSpec, NewSheetData, ResultTable
from ..data_source import SheetData
from ..registry import register
from . import _signal

TASKS = {"spectrum": "see the frequencies", "lowpass": "keep the slow part", "highpass": "keep the fast part",
         "bandpass": "keep a band", "bandstop": "take a band out"}
WINDOWS = {"hann": "Hann", "hamming": "Hamming", "blackman": "Blackman", "none": "none (rectangular)"}


@dataclass
class FourierOptions:
    task: str = "spectrum"           # "spectrum" | "lowpass" | "highpass" | "bandpass" | "bandstop"
    cutoff: float = 0.0              # low-pass and high-pass: the frequency where the filter turns; 0 = a tenth of the highest
    band_low: float = 0.0            # band filters: its two edges; 0, 0 = chosen round the strongest frequency
    band_high: float = 0.0
    sharpness: str = "gentle"        # "gentle" (Butterworth, order 4, no shift in time) | "sharp" (frequencies cut off exactly)
    window: str = "hann"             # for the spectrum
    detrend: str = "mean"            # for the spectrum: "mean" | "line" | "none"
    peaks_listed: int = 5
    create_new_sheet: bool = True
    significant_digits: int = 4


def spectrum(y: np.ndarray, step: float, window: str = "hann", detrend: str = "mean") -> tuple[np.ndarray, np.ndarray]:
    """(frequency, amplitude): a sine wave of amplitude A in the trace shows
    as a peak of height A at its frequency, whatever the window (the
    window's loss of height is made up)."""
    n = y.size
    if detrend == "line":
        y = signal.detrend(y, type="linear")
    elif detrend == "mean":
        y = y - y.mean()
    taper = np.ones(n) if window == "none" else signal.get_window(window, n, fftbins=True)
    amplitude = np.abs(np.fft.rfft(y * taper)) * 2 / taper.sum()
    amplitude[0] /= 2
    if n % 2 == 0:
        amplitude[-1] /= 2
    return np.fft.rfftfreq(n, step), amplitude


def apply_filter(y: np.ndarray, step: float, task: str, low: float, high: float, sharp: bool) -> np.ndarray:
    nyquist = 0.5 / step
    if sharp:
        # every frequency outside what is kept is set to nothing: exact in frequency, but it rings near sudden steps
        freq = np.fft.rfftfreq(y.size, step)
        keep = {"lowpass": freq <= low, "highpass": freq >= low, "bandpass": (freq >= low) & (freq <= high),
                "bandstop": (freq < low) | (freq > high)}[task]
        return np.fft.irfft(np.fft.rfft(y) * keep, n=y.size)
    edges = low / nyquist if task in ("lowpass", "highpass") else [low / nyquist, high / nyquist]
    sos = signal.butter(4, edges, btype={"lowpass": "low", "highpass": "high", "bandpass": "band", "bandstop": "bandstop"}[task],
                        output="sos")
    return signal.sosfiltfilt(sos, y)       # forward and backward: no shift in time, order 8 in effect


@register("fourier_filter")
def run(data: SheetData, options: FourierOptions | None = None) -> AnalysisResult:
    opt = options or FourierOptions()
    task = opt.task if opt.task in TASKS else "spectrum"
    sets = _signal.traces(data, least=16)
    if not sets:
        raise AnalysisError("This needs a trace: numbers in the X column, in steps, and at least 16 rows of values in a data set.")
    sig = opt.significant_digits

    def num(value) -> str:
        return f"{value:.{sig}g}"

    notes: list[str] = []
    prepared = []
    for label, x, y in sets:
        if np.unique(x).size < x.size:
            raise AnalysisError(f"'{label}' has the same X more than once. A trace needs one value at each X.")
        gx, gy, step, moved = _signal.even_grid(x, y)
        if moved:
            notes.append(f"{label}: the X values are not in equal steps, so the trace was read off at {gx.size} equal steps of "
                         f"{num(step)} by straight-line interpolation. Fast changes between uneven points are blurred by this.")
        prepared.append((label, gx, gy, step))

    step = prepared[0][3]
    nyquist = 0.5 / step
    lowest = 1.0 / (prepared[0][1].size * step)
    tables, charts = [], []
    spectra = {label: spectrum(gy, st, opt.window if opt.window in WINDOWS else "hann", opt.detrend) for label, _gx, gy, st in prepared}

    # ── the strongest frequencies of each trace ──────────────────────────────
    rows = []
    strongest = None
    for label, (freq, amp) in spectra.items():
        power = amp**2
        total = power[1:].sum() or 1.0
        tops, _props = signal.find_peaks(np.r_[0.0, amp[1:], 0.0])
        tops = tops[np.argsort(-amp[tops])][:max(1, int(opt.peaks_listed))] if tops.size else np.array([1 + int(np.argmax(amp[1:]))])
        for rank, i in enumerate(tops):
            band = slice(max(1, i - 2), min(amp.size, i + 3))          # a windowed peak is spread over a few points
            rows.append([label if rank == 0 else "", num(freq[i]), num(1 / freq[i]) if freq[i] > 0 else "", num(amp[i]),
                         f"{100 * power[band].sum() / total:.1f}"])
        if strongest is None and tops.size:
            strongest = float(freq[tops[0]])
    tables.append(ResultTable("Strongest frequencies", ["Data set", "Frequency (cycles per unit of X)", "Period (units of X)", "Amplitude",
                                                        "Share of the variation (%)"], rows))
    tables.append(ResultTable("What the sampling allows", ["", "Value"],
                              [["Step in X", num(step)], ["Points", str(prepared[0][1].size)],
                               ["Highest frequency that can be seen (half the sampling rate)", num(nyquist)],
                               ["Lowest, and the spacing between frequencies (1 / length)", num(lowest)]]))
    for label, (freq, amp) in spectra.items():
        charts.append(ChartSpec(f"{label}: how strong each frequency is", x_label="Frequency (cycles per unit of X)", y_label="Amplitude",
                                series=[ChartSeries("", *_signal.for_graph(freq[1:], amp[1:]), kind="line", color="#1f5aa6")],
                                group="spectrum", option=label))

    if task == "spectrum":
        first = rows[0]
        summary = (f"The strongest rhythm in {prepared[0][0]} has a period of {first[2]} (frequency {first[1]}), amplitude {first[3]}, "
                   f"and carries {first[4]}% of the variation. Frequencies up to {num(nyquist)} can be seen at this sampling.")
        notes += [f"Amplitudes are those of the sine waves that make up the trace. Window: {WINDOWS.get(opt.window, 'Hann')}; "
                  + {"mean": "the mean was taken off first.", "line": "a straight-line trend was taken off first.", "none": "nothing was taken off first."}
                  .get(opt.detrend, "the mean was taken off first."),
                  "A window fades the trace in and out, which stops a rhythm that does not fit a whole number of times into the "
                  "record from leaking across the whole spectrum; the price is that each peak is a few points wide.",
                  "A rhythm faster than half the sampling rate cannot be seen as itself: it shows up, falsely, at a lower frequency. "
                  "Nothing in the data can reveal this afterwards.",
                  "Two rhythms closer together than 1 / (length of the record) cannot be told apart.",
                  "Method: fast Fourier transform (Cooley and Tukey 1965); windows after Harris 1978."]
        result = AnalysisResult(title="Frequencies in a trace", summary=summary, tables=tables, notes=notes, charts=charts)
        if opt.create_new_sheet:
            freq = next(iter(spectra.values()))[0]
            same = all(s[0].size == freq.size and np.allclose(s[0], freq) for s in spectra.values())
            if same:
                result.new_sheet = NewSheetData("Frequency spectrum", "Frequency", freq[1:].tolist(),
                                                {label: amp[1:].tolist() for label, (_f, amp) in spectra.items()})
                notes.append("The spectra are on a new sheet, 'Frequency spectrum'.")
            else:
                notes.append("No sheet of spectra was made: the data sets do not share one set of X values.")
        return result

    # ── filtering ────────────────────────────────────────────────────────────
    if task in ("lowpass", "highpass"):
        low = float(opt.cutoff) if opt.cutoff > 0 else nyquist / 10
        high = low
        chosen = "" if opt.cutoff > 0 else " (chosen for you: a tenth of the highest frequency that can be seen)"
        if not 0 < low < nyquist:
            raise AnalysisError(f"The turning frequency must lie between 0 and {num(nyquist)}, half the sampling rate of this trace.")
        what = f"{'below' if task == 'lowpass' else 'above'} {num(low)}"
    else:
        low, high = float(opt.band_low), float(opt.band_high)
        chosen = ""
        if low <= 0 and high <= 0 and strongest:
            low, high = strongest * 0.7, min(strongest * 1.3, nyquist * 0.99)
            chosen = " (chosen for you: 30% either side of the strongest frequency)"
        if not 0 < low < high < nyquist:
            raise AnalysisError(f"The band must run from a frequency above 0 to a higher one below {num(nyquist)}, half the sampling rate.")
        what = f"{'between' if task == 'bandpass' else 'outside'} {num(low)} and {num(high)}"
    sharp = opt.sharpness == "sharp"
    rows, filtered = [], {}
    for label, gx, gy, st in prepared:
        if not sharp and gy.size < 30:
            raise AnalysisError(f"'{label}' has {gy.size} points; the gentle filter needs at least 30. Choose the sharp one, or add points.")
        out = apply_filter(gy, st, task, low, high, sharp)
        filtered[label] = (gx, out)
        kept = float(np.var(out) / np.var(gy)) if np.var(gy) > 0 else float("nan")
        rows.append([label, num(float(np.std(gy))), num(float(np.std(out))), f"{kept:.1%}", num(float(np.max(np.abs(gy - out))))])
        charts.insert(len(filtered) - 1, ChartSpec(f"{label}: before and after", x_label="X", y_label="Y", group="filtered", option=label,
                                                   series=[ChartSeries("As measured", *_signal.for_graph(gx, gy), kind="line", color="#b0b7c1"),
                                                           ChartSeries("Filtered", *_signal.for_graph(gx, out), kind="line", color="#1f5aa6")]))
        f2, a2 = spectrum(out, st, opt.window if opt.window in WINDOWS else "hann", opt.detrend)
        freq, amp = spectra[label]
        lines = [{"kind": "v", "value": v, "color": "#d1342f", "style": "--", "width": 0.9} for v in sorted({low, high})]
        charts.append(ChartSpec(f"{label}: frequencies before and after", x_label="Frequency (cycles per unit of X)", y_label="Amplitude",
                                group="spectrum_after", option=label, reference_lines=lines,
                                series=[ChartSeries("As measured", *_signal.for_graph(freq[1:], amp[1:]), kind="line", color="#b0b7c1"),
                                        ChartSeries("Filtered", *_signal.for_graph(f2[1:], a2[1:]), kind="line", color="#1f5aa6")]))
    tables.insert(0, ResultTable("What the filter did", ["Data set", "SD as measured", "SD after", "Share of the variation kept", "Largest change at a point"],
                                 rows))
    tables.insert(1, ResultTable("The filter", ["", "Value"],
                                 [["Kept", f"frequencies {what}{chosen}"],
                                  ["Kind", "sharp: frequencies outside are removed entirely" if sharp else
                                   "gentle: Butterworth, order 4, run forward and backward (no shift in time)"],
                                  ["That is, periods", (f"{'longer' if task == 'lowpass' else 'shorter'} than {num(1 / low)}" if task in ("lowpass", "highpass")
                                                        else f"{'between' if task == 'bandpass' else 'outside'} {num(1 / high)} and {num(1 / low)}")]]))
    summary = (f"{TASKS[task].capitalize()}: frequencies {what} were kept{chosen}. "
               f"{rows[0][0]} keeps {rows[0][3]} of its variation.")
    notes += ["The gentle filter fades frequencies out round the turning point - at that point the amplitude is halved - and leaves "
              "the trace free of ripples. The sharp one cuts exactly but makes the trace ring near any sudden step."
              if not sharp else
              "The sharp filter removes every frequency outside what is kept. Near a sudden step, and at the two ends, the trace "
              "will ring; choose the gentle one if that matters.",
              "The first and last few cycles of the slowest rhythm kept are the least reliable: a filter has no data beyond the ends.",
              "Filtering changes the data. Do statistics on the measured values unless the filter is part of a method stated in advance.",
              "Method: " + ("Fourier transform, frequencies set to zero, inverse transform." if sharp else
                            "Butterworth filter (1930), applied forward and backward so that no time shift is left.")]
    result = AnalysisResult(title="Frequencies in a trace", summary=summary, tables=tables, notes=notes, charts=charts)
    if opt.create_new_sheet:
        grid = np.unique(np.concatenate([gx for gx, _o in filtered.values()]))
        columns = {}
        for label, (gx, out) in filtered.items():
            column = np.full(grid.size, np.nan)
            column[np.searchsorted(grid, gx)] = out
            columns[label] = column.tolist()
        result.new_sheet = NewSheetData("Filtered", "X", grid.tolist(), columns)
        notes.append("The filtered traces are on a new sheet, 'Filtered'. The sheet analysed is unchanged.")
    return result
