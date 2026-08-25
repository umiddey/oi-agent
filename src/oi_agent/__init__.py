"""OI Agent: standalone 24/7 Discord feedback-channel respondent.

Watches Discord channels (and their threads), answers questions by auditing a
local read-only clone of a repository, and posts replies autonomously. There is
no human-approval tier; safety is structural. The service is fully decoupled
from any watched repo.
"""

__version__ = "0.1.0"
