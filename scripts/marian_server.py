#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import argparse
import json

from bottle import request, Bottle, abort

# Note: the MARIAN_BUILD environment variable points to the directory containing libamunmt.so
# Note: libamunmt.so is the output of the `make python` step of the Marian build
marian_build = os.environ.get('MARIAN_BUILD', None)
if marian_build is None:
    marian_build = os.path.dirname(os.path.realpath(__file__)) + '/../build'

sys.path.append(marian_build)
import libamunmt as nmt


app = Bottle()


@app.route('/translate')
def handle_websocket():
        wsock = request.environ.get('wsgi.websocket')
        if not wsock:
            abort(400, 'Expected WebSocket request.')

        while True:
            try:
                message = wsock.receive()
                if message is not None:
                    segments = json.loads(message)['segments']
                    # force potential unicode to str() for boost conversion
                    listSentences = [segment.encode('utf8') for segment in segments]

                    numEle = len(listSentences)
                    if numEle > 0 and listSentences[numEle - 1] == "":
                        del listSentences[numEle - 1]
                    trans = nmt.translate(listSentences)
                    assert len(trans) == 1, 'we only support single inputs for now (we decode one segment at a time)'
                    trans = trans[0]

                    # parse the n-best list output of Marian
                    n_best_outputs = [segment.split(u' ||| ')[1] for segment in trans.split(u'\n')]
                    wsock.send(json.dumps({'segments': n_best_outputs}))
                    
                    # wsock.send(json.dumps({'segments': trans}))
            except WebSocketError:
                break


def parse_args():
    """ parse command arguments """
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", dest="config")
    parser.add_argument('-p', dest="port", default=8080, type=int)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    nmt.init("-c {}".format(args.config))

    from gevent.pywsgi import WSGIServer
    from geventwebsocket import WebSocketError
    from geventwebsocket.handler import WebSocketHandler
    server = WSGIServer(("0.0.0.0", args.port), app,
                        handler_class=WebSocketHandler)
    server.serve_forever()
