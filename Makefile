PYTHON          := python3
DEPENDENCIES    := requirements.txt
.PHONY: all
all:
	#gcc bin/sketch-0.3.7/*.c -lm -o bin/sketch
	$(PYTHON) -m pip install -r $(DEPENDENCIES)
	./install-xmetis.sh
# Ask user for password up-front
.PHONY: sudo
sudo:
	@sudo -v

.PHONY: init
init: sudo
	sudo $(PYTHON) -m pip install -r $(DEPENDENCIES)

.PHONY: lint
lint:
	flake8 mapper tests

.PHONY: install
install: sudo
	sudo $(PYTHON) -m pip install -e .
	sudo mv bin/sketch /usr/local/bin

.PHONY: test
test:
	$(PYTHON) -m pytest --cov=./mapper --cov-report=term --cov-report=html --tb=short

.PHONY: clean
clean:
	-rm -rf mapper/__pycache__
	-rm -rf tests/__pycache__
	-rm -rf htmlcov
