import clarifai2.api.ClarifaiBuilder;
import clarifai2.api.ClarifaiClient;
import clarifai2.api.ClarifaiResponse;
import clarifai2.dto.input.ClarifaiInput;
import clarifai2.dto.input.image.ClarifaiImage;
import clarifai2.dto.model.output.ClarifaiOutput;
import clarifai2.dto.prediction.Concept;
import okhttp3.OkHttpClient;
import org.openimaj.image.DisplayUtilities;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.MBFImage;
import org.openimaj.image.colour.RGBColour;
import org.openimaj.image.typography.hershey.HersheyFont;

import java.io.File;
import java.io.IOException;
import java.util.List;
/**
 * Created by Gulnoza on 2/5/2017.
 */
public class Image
{
    public static void main (String[] args)throws IOException {
        final ClarifaiClient client = new ClarifaiBuilder("F9cC9V05cS0EatvI3vis1c2rVfS4oHhKexo1LODR", "IT2nK-L9NWELp6uIBx3_M_1wjYhDmm3V3PG_UlM_")
                .client(new OkHttpClient()) // OPTIONAL. Allows customization of OkHttp by the user
                .buildSync(); // or use .build() to get a Future<ClarifaiClient>
        client.getToken();

        KeyFrameDetection keyFrameDetection = new KeyFrameDetection();
        Annotation annotation = new Annotation();
        keyFrameDetection.Frames();
        keyFrameDetection.MainFrames();
        annotation.Annotation();
        File file = new File("output\\frames");
        File[] files = file.listFiles();
        for (int i = 1; i < files.length; i++) {
            ClarifaiResponse response = client.getDefaultModels().generalModel().predict()
                    .withInputs(
                            ClarifaiInput.forImage(ClarifaiImage.of(new File("output\\frames\\new" + i + ".jpg")))
                    )
                    .executeSync();

            List<ClarifaiOutput<Concept>> predictions = (List<ClarifaiOutput<Concept>>) response.get();
            if (predictions.isEmpty()) {
                System.out.println("No Predictions");
            } else {
                MBFImage image = ImageUtilities.readMBF(new File("output\\frames\\new" + i + ".jpg"));
                int x = image.getWidth();
                int y = image.getHeight();


                List<Concept> data = predictions.get(0).data();
                for (int j = 0; j < data.size(); j++) {
                    System.out.println(data.get(j).name() + " - " + data.get(j).value());
                    image.drawText(data.get(j).name(), (int) Math.floor(Math.random() * x), (int) Math.floor(Math.random() * y), HersheyFont.ASTROLOGY, 20, RGBColour.RED);
                }
                DisplayUtilities.displayName(image, "videoFrames");

            }
        }
    }
}
