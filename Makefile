py = python

all: run

run:
	$(py) use.py

build:
	$(py) main.py --build
