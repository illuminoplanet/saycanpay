"""Prompts for all envs."""

from saycanpay.prompts.ravens.put_block_in_bowl import PromptRavensPutBlockInBowl
from saycanpay.prompts.ravens.towers_of_hanoi_seq import PromptRavensTowersOfHanoiSeq
from saycanpay.prompts.babyai.babyai_pickup import PromptBabyAIPickup
from saycanpay.prompts.virtualhome.virtualhome import PromptVirtualHome

names = {
    "put-block-in-bowl": PromptRavensPutBlockInBowl,
    "towers-of-hanoi-seq": PromptRavensTowersOfHanoiSeq,
    "babyai-pickup": PromptBabyAIPickup,
    "virtualhome": PromptVirtualHome,
}
