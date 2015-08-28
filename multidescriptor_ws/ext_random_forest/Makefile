CC = g++
CFLAGS = -Wall -g -ggdb
HEADERS_DIR = headers
DSEARCH = -I ${HEADERS_DIR}
OPENCV_FLAGS = `pkg-config --cflags opencv`
OPENCV_LIBS = `pkg-config --libs opencv`

dectree_exe : dectree_bst.o dectree_class.o erf_class.o erf_main_test.o
	${CC} ${CFLAGS} ${DSEARCH} ${OPENCV_FLAGS} dectree_bst.o dectree_class.o erf_class.o erf_main_test.o -o dectree_exe ${OPENCV_LIBS}

dectree_bst.o : dectree_bst.cpp ${HEADERS_DIR}/dectree_bst.h ${HEADERS_DIR}/dectree_node.h
	${CC} ${CFLAGS} ${DSEARCH} -c dectree_bst.cpp

dectree_class.o : dectree_class.cpp ${HEADERS_DIR}/dectree_class.h ${HEADERS_DIR}/dectree_bst.h ${HEADERS_DIR}/dectree_node.h
	${CC} ${CFLAGS} ${DSEARCH} -c dectree_class.cpp

erf_class.o : erf_class.cpp ${HEADERS_DIR}/erf_class.h ${HEADERS_DIR}/dectree_class.h ${HEADERS_DIR}/dectree_bst.h ${HEADERS_DIR}/dectree_node.h
	${CC} ${CFLAGS} ${DSEARCH} -c erf_class.cpp

#remember the rules to compile - write down later	
#erf_test.o : erf_test.cpp ${HEADERS_DIR}/dectree_class.h ${HEADERS_DIR}/dectree_bst.h
#	${CC} ${CFLAGS} ${DSEARCH} -c erf_test.cpp

erf_main_test.o : erf_main_test.cpp ${HEADERS_DIR}/dectree_class.h ${HEADERS_DIR}/dectree_bst.h
	${CC} ${CFLAGS} ${DSEARCH} -c erf_main_test.cpp

clean :
	rm -rf *o dectree_exe

