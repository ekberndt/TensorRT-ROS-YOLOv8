# -----------------------------------------------------------------------------
# This file defines the following phony targets for the make command
# which define recipes to be executed when the target is called by make.
# This make file can be thought of as a set of custom commands that can be
# used in the race_common repository.
# Usage: make <target>
# -----------------------------------------------------------------------------

# Makes all lines single recipe run to a single shell as by default each line
# will run in it's own shell.
.ONESHELL:
SHELL := /bin/bash
.DEFAULT_GOAL := help

# -----------------------------------------------------------------------------
# Target: help
# Description: Displays the help message for the makefile.
# Parameters: None
# Usage:
#	- make help
#   - make
# -----------------------------------------------------------------------------
.PHONY: help
help:
	@echo "Usage: make <target>"
	@echo ""
	@echo "See Makefile for more information on each target."

# -----------------------------------------------------------------------------
# Target: install-opencv-cuda
# Description: Builds and installs OpenCV with CUDA support system-wide using
# 	yolov8's OpenCV install script. This requires the yolov8 repo to have
# 	already been imported with vcs-import.
# Parameters:
# 	- OPENCV_VERSION: Version of OpenCV to build and install (i.e. 4.8.0).
# 	- CUDA_BIN_ARCH: The architecture of GPU being built for. Look for your GPU's
#		Compute Capability" on https://developer.nvidia.com/cuda-gpus (i.e. 8.9)
# Usage: make install-opencv-cuda OPENCV_VERSION=<OPENCV_VERSION>
#	CUDA_BIN_ARCH=<CUDA_BIN_ARCH>
# -----------------------------------------------------------------------------
.PHONY: install-opencv-cuda
install-opencv-cuda:
	source ./src/yolov8/scripts/install_opencv.sh ${OPENCV_VERSION} ${CUDA_BIN_ARCH}

# -----------------------------------------------------------------------------
# Target: uninstall-opencv-cuda
# Description: Uninstalls OpenCV with CUDA support system-wide using yolov8's
# 	OpenCV uninstall script. This requires the yolov8 repo to have already been
#	imported with vcs-import.
# Parameters: None
# Usage: make uninstall-opencv-cuda
# -----------------------------------------------------------------------------
.PHONY: uninstall-opencv-cuda
uninstall-opencv-cuda:
	source ./src/yolov8/scripts/uninstall_opencv.sh
