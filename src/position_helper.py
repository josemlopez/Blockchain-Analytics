"""
This is a simple helpper application for trailing stop strategy when entering in a
trading position.
The idea is to trail the stop position to meet a maximun radio from the current price.
It will print "EXIT EXIT" if the max pain lost is reached.

"""

import asyncio
from binance import AsyncClient, BinanceSocketManager

LONG = 'long'
SHORT = 'short'


async def main():
    client = await AsyncClient.create()
    bm = BinanceSocketManager(client)
    # start any sockets here, i.e a trade socket
    ts = bm.kline_socket(symbol="BTCUSDT")
    v_acc = 0
    q_acc = 0
    vab_acc = 0
    qvab_acc = 0
    res_acc = {}
    max_pain_radio = 200
    entering = True
    position_tendence = LONG
    position_amount = 10000
    position = {'position_price': 0,
                'return_amount': 0,
                'return': 0,
                'max_pain_price': 0,
                'position_tendence': position_tendence,
                # If we are long, our tndence is -1 our stop price is bellow
                'position_tendence_sign': -1 if position_tendence == 'long' else 1}
    # then start receiving messages
    async with ts as tscm:
        while True:
            res = await tscm.recv()
            v = float(res['k']['v'])
            q = float(res['k']['q'])
            vab = float(res['k']['V'])
            qvab = float(res['k']['Q'])
            current_price = float(res['k']['c'])

            if entering:
                # base case
                position['max_pain_price'] = current_price + position['position_tendence_sign'] * max_pain_radio
                entering = False
                current_return = 0
            else:
                current_return = (current_price - position['position_price']) / position['position_price']
                # max pain price needs to be refreshed in case the price is
                if abs(current_return) > max_pain_radio and \
                    ((position['position_tendence'] == LONG and current_price > position['position_price']) or
                     (position['position_tendence'] == SHORT and current_price < position['position_price'])):
                    position['max_pain_price'] = position['max_pain_price'] + current_return

            position['return'] = position['return'] + current_return
            position['return_amount'] = position['return'] * position_amount

            if abs(current_return * position_amount) > max_pain_radio and \
                ((position['position_tendence'] == LONG and current_price < position['position_price']) or
                 (position['position_tendence'] == SHORT and current_price > position['position_price'])):
                print("- EXIT -" * 30)
                return res

            print("---" * 10)
            print(position)
            print("Precio actual:" + str(current_price))
            print("---" * 10)
            print("\n")

            v_acc = v_acc + v
            q_acc = q_acc + q
            vab_acc = vab_acc + vab
            qvab_acc = qvab_acc + qvab
            res_acc = {"v_acc": v_acc, 'q_acc': q_acc, 'vab_acc': vab_acc, 'qvab_acc': qvab_acc}

            print(res_acc)
            print("*" * 30)
            print(res)
            print("\n")

            if res['k']['x'] is True:
                v_acc = 0
                q_acc = 0
                vab_acc = 0
                qvab_acc = 0

            position['position_price'] = current_price
            await client.close_connection()

if __name__ == "__main__":

    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
