"""Python tier-2: pls4all.sklearn.PLSRegression — idiomatic sklearn surface."""
from _common import parse_args, time_runs, emit

from pls4all.sklearn import PLSRegression


def main():
    X, y, nc, runs = parse_args()

    def do_one():
        m = PLSRegression(nc, solver="simpls",
                           center_x=True, scale_x=False,
                           center_y=True, scale_y=False)
        m.fit(X, y)
        _ = m.predict(X)

    rec = time_runs(do_one, runs)
    emit(rec)


if __name__ == "__main__":
    main()
