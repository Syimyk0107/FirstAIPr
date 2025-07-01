from django.shortcuts import render
from .forms import ImageUploadForm
from django.core.files.storage import FileSystemStorage
from .blip_model import get_blip_description

def upload_image(request):
    description = None
    image_url = None
    full_image_url = None

    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            image = form.cleaned_data['image']

            # сохраняем файл
            fs = FileSystemStorage()
            filename = fs.save(image.name, image)
            image_path = fs.path(filename)
            image_url = fs.url(filename)

            # абсолютный URL для поиска похожих
            full_image_url = request.build_absolute_uri(image_url)

            # получаем описание с BLIP
            description = get_blip_description(image_path)
    else:
        form = ImageUploadForm()

    return render(request, 'upload.html', {
        'form': form,
        'description': description,
        'image_url': image_url,
        'full_image_url': full_image_url,
    })

