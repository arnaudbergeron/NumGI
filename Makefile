coverage:
	coverage erase
	coverage run -m pytest
	coverage report
	coverage html
	coverage xml
