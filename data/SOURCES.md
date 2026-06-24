# AutoFlash RAG Data Sources

Engineering/educational sources only; security-bypass material intentionally excluded.

Retrieved date: 2026-06-24

## Acquired Sources

| Local path | Source URL | Description | Retrieved | License |
| --- | --- | --- | --- | --- |
| `data/raw/ecu-diagnostics-flashing-concepts.md` | user-provided pasted text attachment | ECU diagnostics, UDS service flow, flashing sequence, memory blocks, checksums, containers, calibration, and DTC concepts. | 2026-06-24 | not specified |
| `data/raw/udsoncan/index.rst` | https://github.com/pylessard/python-udsoncan/blob/master/doc/source/index.rst | udsoncan documentation index and navigation. | 2026-06-24 | see upstream repository |
| `data/raw/udsoncan/udsoncan-client.rst` | https://github.com/pylessard/python-udsoncan/blob/master/doc/source/udsoncan/client.rst | UDS client API configuration and behavior reference. | 2026-06-24 | see upstream repository |
| `data/raw/udsoncan/udsoncan-connection.rst` | https://github.com/pylessard/python-udsoncan/blob/master/doc/source/udsoncan/connection.rst | udsoncan connection abstraction reference. | 2026-06-24 | see upstream repository |
| `data/raw/udsoncan/udsoncan-exceptions.rst` | https://github.com/pylessard/python-udsoncan/blob/master/doc/source/udsoncan/exceptions.rst | udsoncan exception types and error handling reference. | 2026-06-24 | see upstream repository |
| `data/raw/udsoncan/udsoncan-helper_classes.rst` | https://github.com/pylessard/python-udsoncan/blob/master/doc/source/udsoncan/helper_classes.rst | Helper classes used by udsoncan requests and responses. | 2026-06-24 | see upstream repository |
| `data/raw/udsoncan/udsoncan-intro.rst` | https://github.com/pylessard/python-udsoncan/blob/master/doc/source/udsoncan/intro.rst | Introductory UDS concepts and udsoncan usage overview. | 2026-06-24 | see upstream repository |
| `data/raw/udsoncan/udsoncan-questions_answers.rst` | https://github.com/pylessard/python-udsoncan/blob/master/doc/source/udsoncan/questions_answers.rst | udsoncan questions and answers for diagnostic workflows. | 2026-06-24 | see upstream repository |
| `data/raw/udsoncan/udsoncan-request_response.rst` | https://github.com/pylessard/python-udsoncan/blob/master/doc/source/udsoncan/request_response.rst | UDS request and response object reference. | 2026-06-24 | see upstream repository |
| `data/raw/udsoncan/udsoncan-services.rst` | https://github.com/pylessard/python-udsoncan/blob/master/doc/source/udsoncan/services.rst | UDS service reference for diagnostic client behavior. | 2026-06-24 | see upstream repository |

## Excluded By Scope

The requested VW_Flash files were fetched but removed from `data/raw/` after inspection because they contained exploit/unlock material outside the allowed engineering/educational scope:

| Requested local path | Source URL | Reason |
| --- | --- | --- |
| `data/raw/vwflash/simos18-architecture.md` | https://raw.githubusercontent.com/bri3d/VW_Flash/master/docs/docs.md | Excluded by scope: exploit/SBOOT material present. |
| `data/raw/vwflash/vwflash-cli.md` | https://raw.githubusercontent.com/bri3d/VW_Flash/master/docs/cli.md | Excluded by scope: unlock/patch workflow material present. |
| `data/raw/udsoncan/udsoncan-examples.rst` | https://github.com/pylessard/python-udsoncan/blob/master/doc/source/udsoncan/examples.rst | Excluded by scope: seed/key algorithm example content present. |

## Manual Downloads (not automatable)

1. CSS Electronics "CAN Bus - The Ultimate Guide" (free 100-page PDF, email-gated): https://www.csselectronics.com/pages/can-bus-ultimate-guide -> save as `data/raw/css-canbus-ultimate-guide.pdf`
2. Wikipedia "OBD-II PIDs" (DTC/PID tables): https://en.wikipedia.org/wiki/OBD-II_PIDs -> use the browser's Print > Save as PDF into `data/raw/obd2-pids-wikipedia.pdf`
