#!/usr/bin/env python
# encoding: utf-8
"""
salami.py
Brian Whitman
brian@echonest.com
"""

import sys
import re
import os, json
# from chord import Chord

# Basedir you put the chord files in 
_base = "/Users/hvkoops/datasets/McGill-Billboard_Salami/"


def parse_salami_file(number):
    """
        Parse a salami_chords.txt file and return a dict with all the stuff innit
    """
    fn = _base + number + "/salami_chords.txt"
    s = open(fn).read().split('\n')
    o = {}
    o["id"] = number
    o["audio_filename"] = _base+number+"/"+number+".mp4"
    # o["en_analysis"] = json.load(open(_base+number+"/echonest.json"))
    for x in s:
        if x.startswith("#"):
            if x[2:].startswith("title:"):
                o["title"] = x[9:]
            if x[2:].startswith("artist:"):
                o["artist"] = x[10:]
            if x[2:].startswith("metre:"):
                o["meter"] = o.get("meter",[]) + [x[9:]]
            if x[2:].startswith("tonic:"):
                o["tonic"] = o.get("tonic",[]) + [x[9:]]
        elif len(x) > 1:
            spot = x.find('\t')
            if spot>0:
                time = float(x[0:spot])
                event = {}
                event["time"] = time
                rest = x[spot+1:]
                items = rest.split(', ')
                for i in items:
                    chords = re.findall(r"(?=\| (.*?) \|)", i)
                    if len(chords):
                        event["chords"] = chords
                    else:
                        event["notes"] = event.get("notes", []) + [i]
                o["events"] = o.get("events", []) + [event]
    return o

def timed_chords(parsed):
    """
        Given a salami parse return a list of parsed chords with timestamps & deltas
    """
    timed_chords = []
    tic = 0
    for i,e in enumerate(parsed["events"]):
        chords = []
        try:
            dt = parsed["events"][i+1]["time"] - e["time"]
        except IndexError:
            dt = 0

        for c in e.get("chords", []):
            for chord_string in c.split(" "): # TODO: figure the difference between | | and spaces
                if chord_string == ".": 
                    chords.append(chords[-1])
                else:
                    chords.append(chord_string)

        tic = e["time"]
        if (len(chords)):
            seconds_per_chord = dt / float(len(chords))
            for c in chords:
                timed_chords.append( {"time":tic, "chord":c, "length":seconds_per_chord} )
                tic = tic + seconds_per_chord
    return timed_chords

def main():
    # Test
    for id_string in open(_base+'all_ids').read().split('\n')[:-1]:
        print( id_string)
        o = parse_salami_file(id_string)
        print (str(o))
    
if __name__ == '__main__':
    main()
