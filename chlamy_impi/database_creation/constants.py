def get_possible_frame_numbers() -> set:
    """The possible numbers of frames in an image array.
    """
    return {84, 92, 100, 164, 172, 180}


def get_time_regime_to_expected_intervals() -> dict[str, set[tuple]]:
    """For each time regime, we have a set of expected time intervals between measurements.
    The intervals are in seconds, found via painful empirical observation.
    """
    return {
        '30s-30s': {(29., 43.), (570, 605), (1760., 1861.)},
        '1min-1min': {(59., 76.), (540, 558), (1730., 1860.)},
        '1min-5min': {(59., 70.), (290., 313.), (420, 436), (540., 555.), (780, 798), (1500, 1528), (1730., 1860.)},
        '5min-5min': {(290., 313.), (1500, 1524), (1730., 1860.)},
        '10min-10min': {(300, 313), (420, 431), (590., 613.), (780, 785), (1500, 1520), (1730., 1860.), (1190., 1220.)},
        '2h-2h': {(1730., 1860.), (599, 607)},
        '20h_ML': {(1730., 1860.), (600, 603)},
        '20h_HL': {(1730., 1860.), (599, 607)}
    }
