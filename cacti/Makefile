# Compiler and flags
CC = g++
OPT_FLAGS = -O3
DBG_FLAGS = -g -O0
WARN_FLAGS = -Wall -Wno-unknown-pragmas
CXXFLAGS = -std=c++11 $(WARN_FLAGS)

# Directories
OBJDIR = obj
OBJDIR_DBG = obj_dbg

# Source files
SRCS = area.cc basic_circuit.cc cacti_interface.cc decoder.cc htree2.cc \
       technology.cc uca.cc wire.cc bank.cc component.cc crossbar.cc \
       io.cc mat.cc router.cc ucache.cc xmlParser.cc

# Object files
OBJS = $(SRCS:%.cc=$(OBJDIR)/%.o)
OBJS_DBG = $(SRCS:%.cc=$(OBJDIR_DBG)/%.o)

# Targets
all: opt

opt: $(OBJDIR) cacti
dbg: $(OBJDIR_DBG) cacti_dbg

$(OBJDIR):
	mkdir -p $(OBJDIR)

$(OBJDIR_DBG):
	mkdir -p $(OBJDIR_DBG)

cacti: $(OBJS)
	$(CC) $(CXXFLAGS) $(OPT_FLAGS) -o $@ $(OBJS)

cacti_dbg: $(OBJS_DBG)
	$(CC) $(CXXFLAGS) $(DBG_FLAGS) -o $@ $(OBJS_DBG)

$(OBJDIR)/%.o: %.cc
	$(CC) $(CXXFLAGS) $(OPT_FLAGS) -c $< -o $@

$(OBJDIR_DBG)/%.o: %.cc
	$(CC) $(CXXFLAGS) $(DBG_FLAGS) -c $< -o $@

clean:
	rm -rf $(OBJDIR) $(OBJDIR_DBG) cacti cacti_dbg

.PHONY: all opt dbg clean 