ver=?
export VER=${ver}

release:
	python3 setup.py sdist && twine upload dist/*
