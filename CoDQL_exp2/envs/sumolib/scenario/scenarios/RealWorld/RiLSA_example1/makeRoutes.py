#!/usr/bin/env python
# Eclipse SUMO, Simulation of Urban MObility; see https://eclipse.org/sumo
# Copyright (C) 2012-2018 German Aerospace Center (DLR) and others.
# This program and the accompanying materials
# are made available under the terms of the Eclipse Public License v2.0
# which accompanies this distribution, and is available at
# http://www.eclipse.org/legal/epl-v20.html
# SPDX-License-Identifier: EPL-2.0

# @file    makeRoutes.py
# @author  Daniel Krajzewicz
# @date    2014-10-06
# @version $Id$

from __future__ import print_function

flows = [
    ["nm", [
        ["ms", 159, 9],
        ["me", 59, 9],
        ["mw", 64, 12]
    ]],

    ["wm", [
        ["me", 708, 10],
        ["mn", 80, 14],
        ["ms", 130, 2]
    ]],

    ["em", [
        ["mw", 571, 10],
        ["mn", 57, 9],
        ["ms", 47, 3]
    ]],

    ["sm", [
        ["mn", 154, 2],
        ["me", 49, 2],
        ["mw", 92, 2]
    ]]

]


fd = open("input_flows.flows.xml", "w")
print("<routes>", file=fd)


for s in flows:
    for d in s[1]:
        id = s[0] + '2' + d[0]
        noLKW = int(float(d[1]) * float(d[2]) / 100.)  # hmph, so korrekt?
        noPKW = d[1] - noLKW
        print(('     <flow id="%sPKW" from="%s" to="%s" number="%s" type="PKW" begin="0" end="3600" departPos="base" ' +
               'arrivalPos="-1" departSpeed="max" departLane="best"/>') % (
            id, s[0], d[0], noPKW), file=fd)
        print(('     <flow id="%sLKW" from="%s" to="%s" number="%s" type="LKW" begin="0" end="3600" departPos="base" ' +
               'arrivalPos="-1" departSpeed="max" departLane="best"/>') % (
            id, s[0], d[0], noLKW), file=fd)
    print("", file=fd)

print("</routes>\n", file=fd)
fd.close()
