from django import forms


class QuizForm(forms.Form):
    selected_answer_id = forms.IntegerField(widget=forms.HiddenInput())


class AdminLoginForm(forms.Form):
    email = forms.EmailField(label='Email')
    password = forms.CharField(label='Password', widget=forms.PasswordInput)
