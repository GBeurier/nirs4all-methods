"""External Python: scikit-learn's PLSRegression."""
from _common import parse_args, time_runs, emit

from sklearn.cross_decomposition import PLSRegression


def main():
    X, y, nc, runs = parse_args()

    def do_one():
        # scale=False matches our center_x=True, scale_x=False settings.
        m = PLSRegression(n_components=nc, scale=False)
        m.fit(X, y)
        _ = m.predict(X)

    rec = time_runs(do_one, runs)
    emit(rec)


if __name__ == "__main__":
    main()
