remove:
	rm -rf ./tmp

install:
	@echo Creating virtual environment...
	python3 -m venv tmp
	@echo Installing pkg...
	./tmp/bin/pip3 install numpy matplotlib

# modify the arguments here
start:
	./tmp/bin/python3 main.py att_faces_10

all:
	make remove install start

.PHONY: remove install start all