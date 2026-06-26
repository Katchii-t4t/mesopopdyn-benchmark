#Makefile
# build python module: make all
# (ignore warnings!)

VERSION=18
CC=gcc
SOURCESpop.c= glm_popdyn_intensiv_no_lookuptbl.c
SOURCESneuron.c= glm_netw_sim_0.8.c
CFLAGS=-lm -lgsl -lgslcblas -lfftw3 -O3 -Wall -Wextra -Wno-unused-result


.PHONY : clean all


all: glm_popdyn.so glm_netw_sim.so



glm_popdyn.so : glm_popdyn.o
	$(CC) -shared -Wl,-soname,glm_popdyn_$(VERSION).so -o glm_popdyn_$(VERSION).so glm_popdyn_$(VERSION).o $(CFLAGS) 

glm_popdyn.o : $(SOURCESpop.c)
	$(CC) -c -fPIC $(SOURCESpop.c) -o glm_popdyn_$(VERSION).o $(CFLAGS) 



glm_netw_sim.so : glm_netw_sim.o
	$(CC) -shared -Wl,-soname,glm_netw_sim_$(VERSION).so -o glm_netw_sim_$(VERSION).so glm_netw_sim_$(VERSION).o $(CFLAGS) 

glm_netw_sim.o : $(SOURCESneuron.c)
	$(CC) -c -fPIC $(SOURCESneuron.c) -o glm_netw_sim_$(VERSION).o $(CFLAGS) 



clean :
	rm -vf glm_popdyn_$(VERSION).so glm_popdyn_$(VERSION).o glm_popdyn.pyc glm_netw sim_$(VERSION).so glm netw_sim_$(VERSION).o glm_netw_sim.pyc


