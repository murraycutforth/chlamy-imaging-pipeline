def get_possible_frame_numbers() -> set:
    """The possible numbers of frames in an image array.

    Includes standard counts and slightly-truncated experiments (1-2 measurements short).
    """
    return {82, 84, 90, 92, 98, 100, 160, 162, 164, 172, 178, 180}


def get_time_regime_to_valid_frame_counts() -> dict[str, set[int]]:
    """For each time regime, the set of valid (corrected) frame counts.

    Includes slightly-truncated experiments (1-2 measurements short) and
    newer Phase II experiment variants with additional measurements.
    """
    return {
        '30s-30s':    {160, 162, 164, 172, 178, 180},
        '1min-1min':  {160, 162, 164, 172, 180},
        '10min-10min': {160, 162, 164, 172, 180},
        '2h-2h':      {82, 84, 98, 100},
        '20h_ML':     {82, 84, 92},
        '20h_HL':     {82, 84, 90, 92},
        '1min-5min':  {180},
        '5min-5min':  {180},
    }


def get_time_regime_to_expected_intervals() -> dict[str, set[tuple]]:
    """For each time regime, we have a set of expected time intervals between measurements.
    The intervals are in seconds, found via painful empirical observation.

    The (900., 940.) range is present in all regimes: it represents a ~15-minute dark
    recovery step that occurs before the final measurement in every experiment.
    """
    _dark_recovery = (900., 940.)
    return {
        '30s-30s': {(29., 43.), (570, 620), (1760., 1861.), _dark_recovery},
        '1min-1min': {(59., 76.), (540, 560), (1730., 1860.), _dark_recovery},
        '1min-5min': {(59., 70.), (290., 313.), (420, 436), (540., 555.), (780, 798), (1500, 1528), (1730., 1860.), _dark_recovery},
        '5min-5min': {(290., 313.), (1500, 1524), (1730., 1860.), _dark_recovery},
        '10min-10min': {(300, 313), (420, 431), (590., 620.), (780, 785), (1500, 1520), (1730., 1860.), (1190., 1220.), _dark_recovery},
        '2h-2h': {(1730., 1860.), (599, 607), _dark_recovery},
        '20h_ML': {(1730., 1860.), (599, 610), _dark_recovery},
        '20h_HL': {(1540., 1620.), (1730., 1862.), (5350., 5450.), (599, 607), _dark_recovery},
    }
