ver=?
export VER=${ver}

url=?
export URL=${url}

release:
	python3 setup.py sdist && twine upload dist/*
