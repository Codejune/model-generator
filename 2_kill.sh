#!/bin/bash
operator=model-generator
pid=$(ps -ef | grep '$operator' | awk '{print $2}' | head -n 1)
kill -9 $pid
