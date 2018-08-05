import itertools
import time
import statistics
import sys
import math

default_number = 1
# default_number = 1000000
default_repeat = 7


def monotonic_clock():
    return time.time()


def mean(data):
    """Return the sample arithmetic mean of data."""
    n = len(data)
    if n < 1:
        raise ValueError('mean requires at least one data point')
    return sum(data)/n  # in Python 2 use sum(data)/float(n)


def _ss(data):
    """Return sum of square deviations of sequence data."""
    c = mean(data)
    ss = sum((x-c)**2 for x in data)
    return ss


def stddev(data, ddof=0):
    """Calculates the population standard deviation
    by default; specify ddof=1 to compute the sample
    standard deviation."""
    n = len(data)
    if n < 2:
        raise ValueError('variance requires at least two data points')
    ss = _ss(data)
    pvar = ss/(n-ddof)
    return pvar**0.5


_TIMEDELTA_UNITS = ('sec', 'ms', 'us', 'ns')


def format_timedeltas(values):
    ref_value = abs(values[0])
    for i in range(2, -9, -1):
        if ref_value >= 10.0 ** i:
            break
    else:
        i = -9

    precision = 2 - i % 3
    k = -(i // 3) if i < 0 else 0
    factor = 10 ** (k * 3)
    unit = _TIMEDELTA_UNITS[k]
    fmt = "%%.%sf %s" % (precision, unit)

    return tuple(fmt % (value * factor,) for value in values)


def format_timedelta(value):
    return format_timedeltas((value,))[0]


def format_filesize(size):
    if size < 10 * 1024:
        if size != 1:
            return '%.0f bytes' % size
        else:
            return '%.0f byte' % size

    if size > 10 * 1024 * 1024:
        return '%.1f MB' % (size / (1024.0 * 1024.0))

    return '%.1f kB' % (size / 1024.0)


def format_filesizes(sizes):
    return tuple(format_filesize(size) for size in sizes)


def format_seconds(seconds):
    # Coarse but human readable duration
    if not seconds:
        return '0 sec'

    if seconds < 1.0:
        return format_timedelta(seconds)

    mins, secs = divmod(seconds, 60.0)
    mins = int(mins)
    hours, mins = divmod(mins, 60)
    days, hours = divmod(hours, 24)

    parts = []
    if days:
        parts.append("%.0f day" % days)
    if hours:
        parts.append("%.0f hour" % hours)
    if mins:
        parts.append("%.0f min" % mins)
    if secs and len(parts) <= 2:
        parts.append('%.1f sec' % secs)
    return ' '.join(parts)


def format_number(number, unit=None, units=None):
    plural = (not number or abs(number) > 1)
    if number >= 10000:
        pow10 = 0
        x = number
        while x >= 10:
            x, r = divmod(x, 10)
            pow10 += 1
            if r:
                break
        if not r:
            number = '10^%s' % pow10

    if isinstance(number, int) and number > 8192:
        pow2 = 0
        x = number
        while x >= 2:
            x, r = divmod(x, 2)
            pow2 += 1
            if r:
                break
        if not r:
            number = '2^%s' % pow2

    if not unit:
        return str(number)

    if plural:
        if not units:
            units = unit + 's'
        return '%s %s' % (number, units)
    else:
        return '%s %s' % (number, unit)


def format_integers(numbers):
    return tuple(format_number(number) for number in numbers)


DEFAULT_UNIT = 'second'
UNIT_FORMATTERS = {
    'second': format_timedeltas,
    'byte': format_filesizes,
    'integer': format_integers,
}


def format_values(unit, values):
    if not unit:
        unit = DEFAULT_UNIT
    formatter = UNIT_FORMATTERS[unit]
    return formatter(values)


def format_value(unit, value):
    return format_values(unit, (value,))[0]


def format_datetime(dt, microsecond=True):
    if not microsecond:
        dt = dt.replace(microsecond=0)
    return dt.isoformat(' ')


def median_abs_dev(values):
    # Median Absolute Deviation
    median = float(statistics.median(values))
    return statistics.median([abs(median - sample) for sample in values])


def percentile(values, p):
    if not isinstance(p, float) or not(0.0 <= p <= 1.0):
        raise ValueError("p must be a float in the range [0.0; 1.0]")

    values = sorted(values)
    if not values:
        raise ValueError("no value")

    k = (len(values) - 1) * p
    # Python 3 returns integers: cast explicitly to int
    # to get the same behaviour on Python 2
    f = int(math.floor(k))
    c = int(math.ceil(k))
    if f != c:
        d0 = values[f] * (c - k)
        d1 = values[c] * (k - f)
        return d0 + d1
    else:
        return values[int(k)]


MAX_LOOPS = 2 ** 32

# Parameters to calibrate and recalibrate warmups

MAX_WARMUP_VALUES = 300
WARMUP_SAMPLE_SIZE = 20


class WorkerTask:
    def __init__(self, name, task_func, args):
        name = name.strip()
        if not name:
            raise ValueError("benchmark name must be a non-empty string")

        self.name = name
        self.args = args
        self.task_func = task_func
        self.loops = args.loops

        self.metadata = dict(unit='second')

        self.inner_loops = None
        self.warmups = None
        self.values = ()

    def _compute_values(self, values, nvalue,
                        is_warmup=False,
                        calibrate_loops=False,
                        start=0):
        unit = self.metadata.get('unit')
        args = self.args
        if nvalue < 1:
            raise ValueError("nvalue must be >= 1")
        if self.loops <= 0:
            raise ValueError("loops must be >= 1")

        if is_warmup:
            value_name = 'Warmup'
        else:
            value_name = 'Value'

        index = 1
        inner_loops = self.inner_loops
        if not inner_loops:
            inner_loops = 1
        while True:
            if index > nvalue:
                break

            raw_value = self.task_func(range(self.loops), monotonic_clock)
            raw_value = float(raw_value)
            value = raw_value / (self.loops * inner_loops)

            if not value and not calibrate_loops:
                raise ValueError("benchmark function returned zero")

            if is_warmup:
                values.append((self.loops, value))
            else:
                values.append(value)

            if args.verbose:
                text = format_value(unit, value)
                if is_warmup:
                    text = ('%s (loops: %s, raw: %s)'
                            % (text,
                               format_number(self.loops),
                               format_value(unit, raw_value)))
                print("%s %s: %s" % (value_name, start + index, text))

            if calibrate_loops and raw_value < args.min_time:
                if self.loops * 2 > MAX_LOOPS:
                    print("ERROR: failed to calibrate the number of loops")
                    print("Raw timing %s with %s is still smaller than "
                          "the minimum time of %s"
                          % (format_value(unit, raw_value),
                             format_number(self.loops, 'loop'),
                             format_timedelta(args.min_time)))
                    sys.exit(1)
                self.loops *= 2
                # need more values for the calibration
                nvalue += 1

            index += 1

    def test_calibrate_warmups(self, nwarmup, unit):
        half = nwarmup + (len(self.warmups) - nwarmup) // 2
        sample1 = [value for loops, value in self.warmups[nwarmup:half]]
        sample2 = [value for loops, value in self.warmups[half:]]
        first_value = sample1[0]

        # test if the first value is an outlier
        values = sample1[1:] + sample2
        q1 = percentile(values, 0.25)
        q3 = percentile(values, 0.75)
        iqr = q3 - q1
        outlier_max = (q3 + 1.5 * iqr)
        # only check maximum, not minimum
        outlier = not(first_value <= outlier_max)

        mean1 = mean(sample1)
        mean2 = mean(sample2)
        mean_diff = (mean1 - mean2) / float(mean2)

        s1_q1 = percentile(sample1, 0.25)
        s2_q1 = percentile(sample2, 0.25)
        s1_q3 = percentile(sample1, 0.75)
        s2_q3 = percentile(sample2, 0.75)
        q1_diff = (s1_q1 - s2_q1) / float(s2_q1)
        q3_diff = (s1_q3 - s2_q3) / float(s2_q3)

        mad1 = median_abs_dev(sample1)
        mad2 = median_abs_dev(sample2)
        # FIXME: handle division by zero
        mad_diff = (mad1 - mad2) / float(mad2)

        if self.args.verbose:
            stdev1 = stddev(sample1)
            stdev2 = stddev(sample2)
            stdev_diff = (stdev1 - stdev2) / float(stdev2)

            sample1_str = format_values(
                unit, (s1_q1, mean1, s1_q3, stdev1, mad1))
            sample2_str = format_values(
                unit, (s2_q1, mean2, s2_q3, stdev2, mad2))
            print("Calibration: warmups=%s" % format_number(nwarmup))
            print("  first value: %s, outlier? %s (max: %s)"
                  % (format_value(unit, first_value), outlier,
                     format_value(unit, outlier_max)))
            print("  sample1(%s): Q1=%s mean=%s Q3=%s stdev=%s MAD=%s"
                  % (len(sample1),
                     sample1_str[0],
                     sample1_str[1],
                     sample1_str[2],
                     sample1_str[3],
                     sample1_str[4]))
            print("  sample2(%s): Q1=%s mean=%s Q3=%s stdev=%s MAD=%s"
                  % (len(sample2),
                     sample2_str[0],
                     sample2_str[1],
                     sample2_str[2],
                     sample2_str[3],
                     sample2_str[4]))
            print("  diff: Q1=%+.0f%% mean=%+.0f%% Q3=%+.0f%% stdev=%+.0f%% MAD=%+.0f%%"
                  % (q1_diff * 100,
                     mean_diff * 100,
                     q3_diff * 100,
                     stdev_diff * 100,
                     mad_diff * 100))

        if outlier:
            return False
        if not(-0.5 <= mean_diff <= 0.10):
            return False
        if abs(mad_diff) > 0.10:
            return False
        if abs(q1_diff) > 0.05:
            return False
        if abs(q3_diff) > 0.05:
            return False
        return True

    def calibrate_warmups(self):
        # calibrate the number of warmups
        if self.loops < 1:
            raise ValueError("loops must be >= 1")

        if self.args.recalibrate_warmups:
            nwarmup = self.args.warmups
        else:
            nwarmup = 1

        unit = self.metadata.get('unit')
        start = 0
        # test_calibrate_warmups() requires at least 2 values per sample
        while True:
            total = nwarmup + WARMUP_SAMPLE_SIZE * 2
            nvalue = total - len(self.warmups)
            if nvalue:
                self._compute_values(self.warmups, nvalue,
                                     is_warmup=True,
                                     start=start)
                start += nvalue

            if self.test_calibrate_warmups(nwarmup, unit):
                break

            if len(self.warmups) >= MAX_WARMUP_VALUES:
                print("ERROR: failed to calibrate the number of warmups")
                values = [format_value(unit, value)
                          for loops, value in self.warmups]
                print("Values (%s): %s" % (len(values), ', '.join(values)))
                sys.exit(1)
            nwarmup += 1

        if self.args.verbose:
            print("Calibration: use %s warmups" % format_number(nwarmup))
            print()

        if self.args.recalibrate_warmups:
            self.metadata['recalibrate_warmups'] = nwarmup
        else:
            self.metadata['calibrate_warmups'] = nwarmup

    def calibrate_loops(self):
        args = self.args
        if not args.recalibrate_loops:
            self.loops = 1

        if args.warmups is not None:
            nvalue = args.warmups
        else:
            nvalue = 1
        nvalue += args.values
        self._compute_values(self.warmups, nvalue,
                             is_warmup=True,
                             calibrate_loops=True)

        if args.verbose:
            print()
            print("Calibration: use %s loops" % format_number(self.loops))
            print()

        if args.recalibrate_loops:
            self.metadata['recalibrate_loops'] = self.loops
        else:
            self.metadata['calibrate_loops'] = self.loops

    def compute_warmups_values(self):
        args = self.args
        if args.warmups:
            self._compute_values(self.warmups, args.warmups, is_warmup=True)
            if args.verbose:
                print()

        self._compute_values(self.values, args.values)
        if args.verbose:
            print()

    def compute(self):
        args = self.args

        self.metadata['name'] = self.name
        if self.inner_loops is not None:
            self.metadata['inner_loops'] = self.inner_loops
        self.warmups = []
        self.values = []

        if args.calibrate_warmups or args.recalibrate_warmups:
            self.calibrate_warmups()
        elif args.calibrate_loops or args.recalibrate_loops:
            self.calibrate_loops()
        else:
            self.compute_warmups_values()

        # collect metatadata
        self.metadata['loops'] = self.loops

    def create_run(self):
        start_time = monotonic_clock()
        self.metadata = dict()
        self.compute()
        self.metadata['duration'] = monotonic_clock() - start_time

        return Run(self.values,
                   warmups=self.warmups,
                   metadata=self.metadata,
                   collect_metadata=False)

    def _set_memory_value(self, value):
        is_calibration = (not self.values)
        self.metadata['unit'] = 'byte'
        self.metadata['warmups'] = len(self.warmups)
        self.metadata['values'] = len(self.values)
        if is_calibration:
            values = ((self.loops, value),)
            self.warmups = values
            self.values = ()
        else:
            self.warmups = None
            self.values = (value,)


class Run:
    def __init__(self, values, warmups, metadata, collect_metadata):
        self.values = values
        self.warmups = warmups
        self.metadata = metadata
        self.collect_metadata = collect_metadata


class Runner:
    def __init__(self):
        self.loops = 0
        self.min_time = 0.1
        self.verbose = False
        self.calibrate_warmups = False
        self.recalibrate_warmups = False
        self.calibrate_loops = False
        self.recalibrate_loops = False
        self.warmups = None
        self.values = 10


def run(inner, verbose=False, loops=None, warmups=None):
    if loops is None:
        runner = Runner()
        runner.calibrate_loops = True
        runner.verbose = verbose
        task = WorkerTask('fannkuch', inner, runner)
        run = task.create_run()
        loops = run.metadata['loops']

    if not verbose:
        print("Calibrated {} loops".format(loops))

    if warmups is None:
        runner = Runner()
        runner.calibrate_warmups = True
        runner.verbose = verbose
        runner.loops = loops
        task = WorkerTask('fannkuch', inner, runner)
        run = task.create_run()
        warmups = run.metadata['calibrate_warmups']

        if not verbose:
            print("Calibrated {} warmups".format(warmups))

    runner = Runner()
    runner.loops = loops
    runner.warmups = warmups
    runner.verbose = verbose
    task = WorkerTask('fannkuch', inner, runner)
    run = task.create_run()

    print("Mean: {}, stddev {}".format(format_value('second', mean(
        run.values)), format_value('second', stddev(run.values))))

    # print(inner(range(default_number), time.time))
