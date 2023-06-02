from django import forms


class QuizForm(forms.Form):
    selected_answer_id = forms.IntegerField(widget=forms.HiddenInput())
