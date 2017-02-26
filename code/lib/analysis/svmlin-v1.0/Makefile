CXXC = g++
CFLAGS = -Wall -O3 -fPIC 

all: svmlin
  
svmlin: svmlin.cpp ssl.o
	$(CXXC) $(CFLAGS) svmlin.cpp ssl.o -o svmlin -lm
ssl.o: ssl.cpp ssl.h
	$(CXXC) $(CFLAGS) -c ssl.cpp
clean:
	rm -f *~ ssl.o svmlin
