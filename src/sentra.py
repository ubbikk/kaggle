import json
import sys

from storm.drpc import DRPCClient

DRPC_FUNCTION = "compute_sentiments"
api_host = "10.1.2.117"
api_port = 3772
fp='/home/dpetrovskyi/Desktop/y.txt'


def load_text():
    with open(fp) as f:
        s= f.read()
        s=s[:len(s)-1]
        s='{\"x\" : \"'+s+'\"}'
        return json.loads(s)['x']


def excecute(domain):
    client = DRPCClient(api_host, api_port)
    query = u'[{{"content": {}, "id": "{}", "domain": "{}"}}]'.format(json.dumps(load_text()), 'stub', domain)
    return client.execute(DRPC_FUNCTION, query)

print excecute(sys.argv[1])
