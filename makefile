CXX = g++-8

# path #
SRC_PATH = src
BUILD_PATH = build
BIN_PATH = bin

# executable # 
BIN_NAME = bdd-cuda

# extensions #
SRC_EXT = cpp

# code lists #
# Find all source files in the source directory, sorted by
# most recently modified
SOURCES = $(shell find $(SRC_PATH) -name '*.$(SRC_EXT)' | sort -k 1nr | cut -f2-)

# Set the object file names, with the source directory stripped
# from the path, and the build path prepended in its place
OBJECTS = $(SOURCES:$(SRC_PATH)/%.$(SRC_EXT)=$(BUILD_PATH)/%.o)

# Set the dependency files that will be used to add header dependencies
# DEPS = $(OBJECTS:.o=.d)

# flags #
COMPILE_FLAGS = -fopenmp -Winline -O3 -g -std=c++17 -Wall -Wextra -DYY_TYPEDEF_YY_SIZE_T -Dyy_size_t=ssize_t
INCLUDES = -I include/ -I /usr/local/include
# Space-separated pkg-config libraries used by this project
LIBS = 

.PHONY: default_target
default_target: release

.PHONY: release
release: export CXXFLAGS := $(CXXFLAGS) $(COMPILE_FLAGS)
release: dirs
	@$(MAKE) all

.PHONY: dirs
dirs:
	@echo "Creating directories"
	@mkdir -p $(BUILD_PATH)
	@mkdir -p $(BIN_PATH)


# checks the executable and symlinks to the output
.PHONY: all
all: $(BIN_PATH)/$(BIN_NAME) test_cuda
	@echo "Making symlink: $(BIN_NAME) -> $<"
	@$(RM) $(BIN_NAME)
	@ln -s $(BIN_PATH)/$(BIN_NAME) $(BIN_NAME)

# Creation of the executable
$(BIN_PATH)/$(BIN_NAME): $(OBJECTS)#
	@echo "Linking: $@"
	$(CXX) $(CXXFLAGS) $(OBJECTS) $(LIBS) -o $@

# Creation of the cuda executable
test_cuda: $(BIN_PATH)/$(BIN_NAME)
	@echo "Creating cuda executable: $@"
	nvcc -o $(BIN_PATH)/test_cuda $(SRC_PATH)/test_cuda.cu

# Add dependency files, if they exist
-include $(DEPS)

# Source file rules
# After the first compilation they will be joined with the rules from the
# dependency files to provide header dependencies

$(BUILD_PATH)/%.o: $(SRC_PATH)/%.$(SRC_EXT)
	@echo "Compiling: $< -> $@"
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@



.PHONY: clean
clean:
	@echo "Deleting $(BIN_NAME) symlink"
	@$(RM) $(BIN_NAME)
	@echo "Deleting directories"
	@$(RM) -r $(BUILD_PATH)
	@$(RM) -r $(BIN_PATH)



