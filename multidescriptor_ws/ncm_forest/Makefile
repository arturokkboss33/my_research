CC = g++
CFLAGS = -Wall -g -ggdb
HEADERS_DIR = headers
DSEARCH = -I ${HEADERS_DIR}
OPENCV_FLAGS = `pkg-config --cflags opencv`
OPENCV_LIBS = `pkg-config --libs opencv`

#dectree_exe : dectree_bst.o dectree_class.o erf_class.o erf_test.o
#	${CC} ${CFLAGS} ${DSEARCH} ${OPENCV_FLAGS} dectree_bst.o dectree_class.o erf_class.o erf_test.o -o dectree_exe ${OPENCV_LIBS}

#dectree_bst.o : dectree_bst.cpp ${HEADERS_DIR}/dectree_bst.h ${HEADERS_DIR}/dectree_node.h
#	${CC} ${CFLAGS} ${DSEARCH} -c dectree_bst.cpp

#dectree_class.o : dectree_class.cpp ${HEADERS_DIR}/dectree_class.h ${HEADERS_DIR}/dectree_bst.h ${HEADERS_DIR}/dectree_node.h
#	${CC} ${CFLAGS} ${DSEARCH} -c dectree_class.cpp

#erf_class.o : erf_class.cpp ${HEADERS_DIR}/erf_class.h ${HEADERS_DIR}/dectree_class.h ${HEADERS_DIR}/dectree_bst.h ${HEADERS_DIR}/#dectree_node.h
#	${CC} ${CFLAGS} ${DSEARCH} -c erf_class.cpp

#remember the rules to compile - write down later	
#erf_test.o : erf_test.cpp ${HEADERS_DIR}/dectree_class.h ${HEADERS_DIR}/dectree_bst.h
#	${CC} ${CFLAGS} ${DSEARCH} -c erf_test.cpp

ncmf_exe : ncmf_bst.o ncmf_class_tree.o ncmf_forest.o ncmf_test.o
	${CC} ${CFLAGS} ${DSEARCH} ${OPENCV_FLAGS} ncmf_bst.o ncmf_class_tree.o ncmf_forest.o ncmf_test.o -o ncmf_exe ${OPENCV_LIBS}

ncmf_bst.o : ncmf_bst.cpp ${HEADERS_DIR}/ncmf_bst.h ${HEADERS_DIR}/ncmf_node.h
	${CC} ${CFLAGS} ${DSEARCH} -c ncmf_bst.cpp

ncmf_class_tree.o : ncmf_class_tree.cpp ${HEADERS_DIR}/ncmf_class_tree.h ${HEADERS_DIR}/ncmf_bst.h ${HEADERS_DIR}/ncmf_node.h
	${CC} ${CFLAGS} ${DSEARCH} -c ncmf_class_tree.cpp

ncmf_forest.o : ncmf_forest.cpp ${HEADERS_DIR}/ncmf_forest.h ${HEADERS_DIR}/ncmf_class_tree.h
	${CC} ${CFLAGS} ${DSEARCH} -c ncmf_forest.cpp

ncmf_test.o : ncmf_test.cpp ${HEADERS_DIR}/ncmf_class_tree.h ${HEADERS_DIR}/ncmf_bst.h ${HEADERS_DIR}/ncmf_node.h
	${CC} ${CFLAGS} ${DSEARCH} -c ncmf_test.cpp

clean :
	rm -rf *o dectree_exe

