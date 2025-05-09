# test_summarization.py

import torch
from transformers import BartTokenizer, BartForConditionalGeneration

def load_model(model_name="facebook/bart-large-cnn", device="cpu"):
    """
    모델과 토크나이저를 로드합니다.
    """
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model     = BartForConditionalGeneration.from_pretrained(model_name).to(device)
    model.eval()
    return tokenizer, model

def summarize(texts, tokenizer, model, device="cpu", max_length=150, min_length=40):
    """
    텍스트 리스트를 받아 요약문을 반환합니다.
    """
    # 배치 형태로 토크나이즈
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)
    # 요약 생성
    summary_ids = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=max_length,
        min_length=min_length,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
    )
    # 디코딩
    summaries = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                 for g in summary_ids]
    return summaries

def main():
    # GPU 사용 가능 시 GPU로
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer, model = load_model(device=device)

    # 테스트할 긴 텍스트
    raw = """Supreme Court Chief Justice John Roberts used a public appearance Wednesday to stress the importance of an independent judiciary.
Supreme Court Chief Justice John Roberts used a public appearance Wednesday to stress the importance of an independent judiciary, doubling down on defense of the courts under fire by PresidentDonald Trumpand his allies, who have accused so-called "activist judges" of overstepping their bounds.
Asked during a fireside chat event in Buffalo, New York, about judicial independence, Roberts responded in no uncertain terms that the role of the federal courts is to "decide cases, but in the course of that, check the excesses of Congress or the executive."
That role, he added, "does require a degree of independence."
BOASBERG GRILLS DOJ OVER REMARKS FROM TRUMP AND NOEM, FLOATS MOVING MIGRANTS TO GITMO IN ACTION-PACKED HEARING
President Donald Trump shakes hands with Supreme Court Chief Justice John Roberts as Melania Trump and family look on at the U.S. Capitol Jan. 20, 2025, in Washington, D.C.(Chip Somodevilla/Pool via Reuters)
Roberts' remarks are not new. But they comeas Trump and his allies haverailed against federal judges who have paused or halted key parts of the president's agenda. (Some of the rulings they've taken issue with came from judges appointed by Trump in his first term.)
The Supreme Court is slated to hear a number of high-profile cases and emergency appeals filed by the Trump administration in the next few months, cases that are all but certain to keep the high court in the spotlight for the foreseeable future.
Among them are Trump's executive orders banning transgender service members from serving in the U.S. military, restoring fired federal employees to their jobs and a case about whether children whose parents illegally entered the U.S. and were born here should be granted citizenship. Oral arguments for that last case kick off next week.
TRUMP-ALIGNED GROUP SUES CHIEF JUSTICE JOHN ROBERTS IN EFFORT TO RESTRICT POWER OF THE COURTS
Chief Justice John Roberts, right, speaks with U.S. District Judge Lawrence J. Vilardo during a fireside chat in Buffalo, N.Y.(AP Photo/Jeffrey T. Barnes)
Just hours before Roberts spoke to U.S. District Judge Lawrence Vilardo, a high-stakes hearing played out in federal court in Washington, D.C.
There, U.S. District Judge James Boasbergspent more than an hourgrilling Justice Department lawyers about their use of the Alien Enemies Act to summarily deport hundreds of migrants to El Salvador earlier this year.
Boasberg’s March 15 order that temporarily blocked Trump’s use of the law to send migrants to a Salvadoran prison sparked ire from the White House and in Congress, where some Trump allies had previously floated calls for impeachment.
Chief Justice John Roberts speaks during a fireside chat at the 125th anniversary celebration of the United States District Court for the Western District of New York Wednesday, May 7, 2025, in Buffalo, N.Y.(AP Photo/Jeffrey T. Barnes)
Roberts, who put out a rare public statement at the time rebuking calls to impeach Boasberg or any federal judges, doubled down on that in Wednesday's remarks.
"Impeachment is not how you register disagreement with a decision," Roberts said, adding that he had already spoken about that in his earlier statement.
CLICK HERE TO GET THE FOX NEWS APP
In the statement, sent by Roberts shortly after Trump floated the idea of impeaching Boasberg, said that "for more than two centuries, it has been established that impeachment is not an appropriate response to disagreement concerning a judicial decision," he said.
"The normal appellate review process exists for that purpose," he said in the statement.
Breanne Deppisch is a national politics reporter for Fox News Digital covering the Trump administration, with a focus on the Justice Department, FBI, and other national news."""

    # 한 덩어리로 요약
    summaries = summarize([raw], tokenizer, model, device=device)
    print("Original Text:\n", raw, "\n")
    print("Summary:\n", summaries[0])

if __name__ == "__main__":
    main()
