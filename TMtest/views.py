from django.shortcuts import render,redirect
from .forms import DataForm
from .models import Temper
# Create your views here.
def test(request):
    print(request.method)
    if request.method == 'POST':
        print('****')
        form = DataForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('result')
    else:
        form = DataForm()
    context = {
        'form': form
    }
    return render(request, 'TMtest/test.html', context)

def result(request):
    last_temper=Temper.objects.last()
    context={
        'temper':last_temper
    }
    return render(request,'TMtest/result.html',context)