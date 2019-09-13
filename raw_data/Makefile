correlate_pulse: correlate_pulse.c
	gcc -o $@ $^ -I/usr/local/include -L/usr/local/lib -lcfitsio -pthread -lm -O3

make_filterbank: make_filterbank.c
	gcc -o $@ $^ -I/usr/local/include -I/usr/local/sigproc/src -L/usr/local/lib -L/usr/local/sigproc/src -lsigproc -lcfitsio -pthread -lm -O3

correlate_pulse_calib: correlate_pulse_calib.c
	gcc -o $@ $^ -I/usr/local/include -L/usr/local/lib -lcfitsio -pthread -lm -O3

correlate_pulse_tmp: correlate_pulse_tmp.c
	gcc -o $@ $^ -I/usr/local/include -L/usr/local/lib -lcfitsio -pthread -lm -O3

correlate_pulse_bp: correlate_pulse_bp.c
	gcc -o $@ $^ -I/usr/local/include -L/usr/local/lib -lcfitsio -pthread -lm -O3

correlate_pulse_bp_ri: correlate_pulse_bp_ri.c
	gcc -o $@ $^ -I/usr/local/include -L/usr/local/lib -lcfitsio -pthread -lm -O3

acorr_raw: acorr_raw.c
	gcc -o $@ $^ -I/usr/local/include -L/usr/local/lib -lcfitsio -pthread -lm -O3

